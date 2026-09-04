#!/usr/bin/env bash
# =============================================================================
#  run_model.sh - Linux llama-server launcher (counterpart of run_model.bat)
#
#  Menu flow: model -> backend -> vision -> context -> MTP -> launch
#  Backends are auto-discovered from backends/bin/* (any dir containing a
#  working `llama-server`). Family is inferred from the dir name:
#    *rocm* / *hip*  -> ROCm configs
#    otherwise       -> Vulkan configs
#
#  Windows-only features from the .bat (GPU reset, ADLX power profile) are
#  intentionally omitted.
#
#  Usage:
#    bash run_model.sh              interactive menu
#    bash run_model.sh --dry-run    pick options, print command, don't launch
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Symlink in the repo (models -> external LM Studio folder) so paths stay portable.
MODELS_ROOT="$ROOT/models"
PORT=8081
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

SAMPLING_FLAG="--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0"

# ── Model files ──────────────────────────────────────────────────────────────
MODEL_IQ="$MODELS_ROOT/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
MODEL_KATXS="$MODELS_ROOT/thread13/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF-MTP/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.bartowski.mtp.gguf"
MODEL_Q38="$MODELS_ROOT/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf"
MODEL_Q38M="$MODELS_ROOT/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_M.gguf"
MODEL_Q3X="$MODELS_ROOT/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf"
MODEL_Q2X="$MODELS_ROOT/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf"
MODEL_RIDGE="$MODELS_ROOT/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf"
MODEL_Q8="$MODELS_ROOT/empero-ai/Qwen3.8-9B-Distill-GGUF/Qwen3.8-9B-Q8_0.gguf"
MODEL_GSQ_RCO="$MODELS_ROOT/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
DFLASH_DRAFT="$MODELS_ROOT/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

MMPROJ_Q36="$MODELS_ROOT/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F32.gguf"
MMPROJ_Q38="$MODELS_ROOT/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf"
MMPROJ_RIDGE="$MODELS_ROOT/empero-ai/Qwen3.8-27B-Ridge-GGUF/mmproj-Qwen3.8-27B-BF16.gguf"

# ── Discover backends ────────────────────────────────────────────────────────
# Each entry: "name|dir|family"
BACKENDS=()
for _d in "$ROOT"/backends/bin/*; do
    [ -x "$_d/llama-server" ] || continue
    _name="$(basename "$_d")"
    case "$_name" in
        *.bak|*.old) continue ;;  # stale preserved builds (e.g. rocm-linux-b10488.bak)
    esac
    case "$_name" in
        *rocm*|*hip*) _fam="rocm" ;;
        *)            _fam="vk"   ;;
    esac
    BACKENDS+=("$_name|$_d|$_fam")
done
if [ "${#BACKENDS[@]}" -eq 0 ]; then
    echo "No llama-server found under $ROOT/backends/bin/ (run scripts/build_linux_vulkan.sh first)"
    exit 1
fi

# ── State ────────────────────────────────────────────────────────────────────
MODEL=""
BACKEND_NAME=""
SERVER=""
BACKEND_IS_VK=1
MMPROJ_PATH=""
MMPROJ_OFFLOAD=0
CTX_SIZE=32768
MTP_MODE="off"
DFLASH_MODE="off"
SPEC_FLAG=()
Q38_FLAG=""
KV_K="q4_0"; KV_V="q4_0"; KV_SET=0
CACHE_RAM=0
B=2048; UB=2048; NCMOE=0
MOE_BACKEND=0
MOE_CACHE_PROFILE=""
MOE_CACHE_SLOTS=0

# ── Helpers ──────────────────────────────────────────────────────────────────
pick() {  # pick <prompt> <count> -> echoes 1..count
    local prompt="$1" n="$2" ans
    while true; do
        printf '%s ' "$prompt" >&2
        read -r ans
        if [[ "$ans" =~ ^[0-9]+$ ]] && [ "$ans" -ge 1 ] && [ "$ans" -le "$n" ]; then
            echo "$ans"; return 0
        fi
        echo "  invalid choice (1-$n)" >&2
    done
}

header() {
    echo "================================================"
    echo "  Model  : $MODEL"
    echo "  Backend: $SERVER"
    [ -n "$CTX_SIZE" ] && echo "  Context: $CTX_SIZE"
    echo "================================================"
    echo
}

# ── Model selection ──────────────────────────────────────────────────────────
menu_model() {
    KV_K="q4_0"; KV_V="q4_0"; KV_SET=0
    CACHE_RAM=0
    SPEC_FLAG=(); MTP_MODE="off"; DFLASH_MODE="off"
    MMPROJ_PATH=""; MMPROJ_OFFLOAD=0
    Q38_FLAG=""
    clear 2>/dev/null || true
    echo "================================================"
    echo "  Qwen3.6 / KAT-Coder / Qwen3.8 - Launcher (Linux)"
    echo "================================================"
    echo
    echo "  Select model:"
    echo "   [1] Qwen3.6 IQ4_XS MTP (18.2 GB, MTP, vision)"
    echo "   [2] KAT-Coder IQ4_XS MTP (19.7 GB)"
    echo "   [3] Qwen3.8-27B IQ4_XS (14.3 GB, dense, vision)"
    echo "   [4] Qwen3.8-27B Q3_K_XL (13.1 GB, dense, vision)"
    echo "   [5] Qwen3.8-27B Q3_K_M (13.8 GB, dense, vision)"
    echo "   [6] Qwen3.8-27B Ridge-3.7bpw (12.6 GB, dense, vision)"
    echo "   [7] Qwen3.8-27B Q2_K_XL (~9 GB, dense, vision)"
    echo "   [8] Qwen3.8-9B-Distill Q8_0 (9.1 GB, q8_0 KV)"
    echo "   [9] Qwen3.8-27B GSQ-RCO IQ3_S (11.8 GB, dense, vision, mixed-precision quant, DFlash2-tested)"
    echo
    case "$(pick "  Choice [1-9]: " 9)" in
        1) MODEL="$MODEL_IQ";    MMPROJ_PATH="$MMPROJ_Q36" ;;
        2) MODEL="$MODEL_KATXS" ;;
        # reasoning via template kwargs (--jinja) instead of --reasoning-preserve/--reasoning-effort
        3) MODEL="$MODEL_Q38";   MMPROJ_PATH="$MMPROJ_Q38"; Q38_FLAG="--reasoning on --chat-template-kwargs {\"reasoning_effort\":\"medium\",\"preserve_reasoning\":true}" ;;
        4) MODEL="$MODEL_Q3X";   MMPROJ_PATH="$MMPROJ_Q38"; Q38_FLAG="--reasoning on --chat-template-kwargs {\"reasoning_effort\":\"medium\",\"preserve_reasoning\":true}" ;;
        5) MODEL="$MODEL_Q38M";  MMPROJ_PATH="$MMPROJ_Q38"; Q38_FLAG="--reasoning on --chat-template-kwargs {\"reasoning_effort\":\"medium\",\"preserve_reasoning\":true}" ;;
        6) MODEL="$MODEL_RIDGE"; MMPROJ_PATH="$MMPROJ_RIDGE"; Q38_FLAG="--reasoning on --chat-template-kwargs {\"reasoning_effort\":\"medium\",\"preserve_reasoning\":true}" ;;
        7) MODEL="$MODEL_Q2X";   MMPROJ_PATH="$MMPROJ_Q38"; Q38_FLAG="--reasoning on --chat-template-kwargs {\"reasoning_effort\":\"medium\",\"preserve_reasoning\":true}" ;;
        8) MODEL="$MODEL_Q8";    KV_K="q8_0"; KV_V="q8_0"; Q38_FLAG="--reasoning on --chat-template-kwargs {\"reasoning_effort\":\"medium\",\"preserve_reasoning\":true}" ;;
        9) MODEL="$MODEL_GSQ_RCO"; MMPROJ_PATH="$MMPROJ_Q38"; Q38_FLAG="--reasoning on --chat-template-kwargs {\"reasoning_effort\":\"medium\",\"preserve_reasoning\":true}" ;;
    esac
    [ -f "$MODEL" ] || echo "  WARNING: model file not found: $MODEL"
}

# ── Backend selection ────────────────────────────────────────────────────────
menu_backend() {
    clear 2>/dev/null || true
    header
    echo "  Select backend (auto-discovered):"
    local i=1
    for be in "${BACKENDS[@]}"; do
        local name="${be%%|*}"
        case "$name" in *moe*) extra=" (perf fork - MoE expert cache)" ;; *) extra="" ;; esac
        printf '   [%d] %s%s\n' "$i" "$name" "$extra"
        i=$((i + 1))
    done
    echo
    local sel
    sel="$(pick "  Backend [1-${#BACKENDS[@]}]: " "${#BACKENDS[@]}")"
    local be="${BACKENDS[$((sel - 1))]}"
    BACKEND_NAME="${be%%|*}"
    SERVER="${be#*|}"
    SERVER="${SERVER%%|*}/llama-server"
    case "${be##*|}" in rocm) BACKEND_IS_VK=0 ;; *) BACKEND_IS_VK=1 ;; esac
    case "$BACKEND_NAME" in *moe*) MOE_BACKEND=1 ;; *) MOE_BACKEND=0 ;; esac
}

# ── Vision toggle ────────────────────────────────────────────────────────────
menu_vision() {
    [ -n "$MMPROJ_PATH" ] || { MMPROJ_OFFLOAD=0; return; }
    [ -f "$MMPROJ_PATH" ] || { echo "  (mmproj missing, skipping vision)"; MMPROJ_OFFLOAD=0; return; }
    # stew fork crashes on the vision encoder (MUL_MAT)
    if [ "$BACKEND_NAME" = "stew" ]; then
        echo "  Vision NOT available on stew backend - continuing without"
        MMPROJ_OFFLOAD=0
        return
    fi
    clear 2>/dev/null || true
    header
    echo "  Vision (multimodal projector):"
    echo "   [1] ON - CPU  (--no-mmproj-offload, saves ~0.9 GB VRAM)"
    echo "   [2] ON - GPU  (faster image encode)"
    echo "   [3] OFF"
    echo
    case "$(pick "  Vision [1-3]: " 3)" in
        1) MMPROJ_OFFLOAD=1 ;;
        2) MMPROJ_OFFLOAD=0 ;;
        3) MMPROJ_PATH=""  ;;
    esac
}

# ── Context selection (speeds are Windows-Vulkan baselines; Ridge updated with
#    Linux RADV measurements from 2026-08-28) ────────────────────────────────
set_speeds() {
    PP32="~2900"; PP64="~2500"; PP128="~2400"; PP180="~1200"; PP200="~1200"; PP256="~1200"
    SPD32="~37";  SPD64="~48";  SPD128="~47";  SPD180="~40";  SPD200="~40";  SPD256="~45"
    case "$MODEL" in
        "$MODEL_Q38")
            PP32="~40"; PP64="~40"; PP128="~40"; PP256="❌"
            SPD32="~7";  SPD64="~7";  SPD128="~7";  SPD256="❌" ;;
        "$MODEL_Q38M")
            PP32="~655"; PP64="~646"; PP128="~570"; PP256="❌102"
            SPD32="~37";  SPD64="~32";  SPD128="~35";  SPD256="❌" ;;
        "$MODEL_Q3X")
            PP32="~650"; PP64="~650"; PP128="~650"; PP256="❌"
            SPD32="~33";  SPD64="~33";  SPD128="~33";  SPD256="❌" ;;
        "$MODEL_Q2X")
            PP32="~650"; PP64="~650"; PP128="~650"; PP256="~?"
            SPD32="~45";  SPD64="~45";  SPD128="~45";  SPD256="~?" ;;
        "$MODEL_Q8")
            PP32="~2670"; PP64="~2670"; PP128="~2670"; PP256="~2670"
            SPD32="~58";  SPD64="~58";  SPD128="~58";  SPD256="~58" ;;
        "$MODEL_RIDGE")
            PP32="~1130"; PP64="~? "; PP128="~?"; PP256="~?"
            SPD32="~39";  SPD64="~?";  SPD128="~?";  SPD256="~?" ;;
        "$MODEL_KATXS")
            PP32="~2900"; PP64="~2500"; PP128="~2400"; PP256="~1200"
            SPD32="~53";  SPD64="~53";  SPD128="~53";  SPD256="~53" ;;
        "$MODEL_GSQ_RCO")
            # not yet benchmarked at these context sizes (only DFlash2+ngram sanity-tested so far)
            PP32="~?"; PP64="~?"; PP128="~?"; PP256="~?"
            SPD32="~?"; SPD64="~?"; SPD128="~?"; SPD256="~?" ;;
    esac
    # 80K/100K/150K aren't in the original speed tables - inherit 64K/128K by proximity
    PP80="${PP80:-$PP64}";   SPD80="${SPD80:-$SPD64}"
    PP100="${PP100:-$PP128}"; SPD100="${SPD100:-$SPD128}"
    PP150="${PP150:-$PP128}"; SPD150="${SPD150:-$SPD128}"
}

menu_ctx() {
    clear 2>/dev/null || true
    set_speeds
    header
    echo "  Select context window (pp | tg t/s):"
    echo "   [1] 32K   pp $PP32 | tg $SPD32"
    echo "   [2] 64K   pp $PP64 | tg $SPD64"
    echo "   [3] 80K   pp $PP80 | tg $SPD80"
    echo "   [4] 100K  pp $PP100 | tg $SPD100"
    echo "   [5] 128K  pp $PP128 | tg $SPD128"
    echo "   [6] 150K  pp $PP150 | tg $SPD150"
    echo "   [7] 180K  pp $PP180 | tg $SPD180"
    echo "   [8] 200K  pp $PP200 | tg $SPD200"
    echo "   [9] 256K  pp $PP256 | tg $SPD256"
    echo
    case "$(pick "  Context [1-9]: " 9)" in
        1) CTX_SIZE=32768  ;;
        2) CTX_SIZE=65536  ;;
        3) CTX_SIZE=81920  ;;
        4) CTX_SIZE=102400 ;;
        5) CTX_SIZE=131072 ;;
        6) CTX_SIZE=153600 ;;
        7) CTX_SIZE=184320 ;;
        8) CTX_SIZE=204800 ;;
        9) CTX_SIZE=262144 ;;
    esac
}

# ── KV cache type selection ──────────────────────────────────────────────────
# All types llama.cpp accepts for -ctk/-ctv. A/B notes from the RX 9070 XT:
#   q4_0   safest default
#   q4_1   vulkan-only win (Ridge spec); q5_1 broken on ROCm
#   q8_0   pair with Q8_0 weights
#   iq4_nl i-quant: crashes official ROCm builds (vulkan/selfbuilt/stew OK)
KV_TYPES=(q4_0 q4_1 q5_0 q5_1 q8_0 f16 f32 iq4_nl)

menu_kv() {
    clear 2>/dev/null || true
    header
    echo "  Select K cache type (-ctk, current: $KV_K):"
    local i=1
    for t in "${KV_TYPES[@]}"; do
        printf '   [%d] %s%s\n' "$i" "$t" "$([ "$t" = "$KV_K" ] && echo '  (current)')"
        i=$((i + 1))
    done
    echo
    local k v
    k="$(pick "  K type [1-${#KV_TYPES[@]}]: " "${#KV_TYPES[@]}")"
    clear 2>/dev/null || true
    header
    echo "  Select V cache type (-ctv):"
    echo "   [1] same as K (${KV_TYPES[$((k - 1))]})"
    local i=2
    for t in "${KV_TYPES[@]}"; do
        printf '   [%d] %s%s\n' "$i" "$t" "$([ "$t" = "$KV_V" ] && echo '  (current)')"
        i=$((i + 1))
    done
    echo
    v="$(pick "  V type [1-$(( ${#KV_TYPES[@]} + 1 ))]: " "$(( ${#KV_TYPES[@]} + 1 ))")"
    KV_K="${KV_TYPES[$((k - 1))]}"
    if [ "$v" -eq 1 ]; then
        KV_V="$KV_K"
    else
        KV_V="${KV_TYPES[$((v - 2))]}"
    fi
    KV_SET=1
}

# ── MTP toggle ───────────────────────────────────────────────────────────────
menu_mtp() {
    clear 2>/dev/null || true
    header
    # DFlash2 (draft-dflash) on all Qwen3.8-27B models. rocm-linux/vulkan-linux
    # need a b10701+ build (upstream commit-count build numbers). rocm-stew675-linux
    # tracks its own baseline+patches build numbering (currently 10630, below that
    # threshold) but `--help` confirms draft-dflash is compiled in regardless, so
    # it's exempted from the version-number gate.
    DFLASH_OK=0
    case "$MODEL" in
        "$MODEL_Q38"|"$MODEL_Q38M"|"$MODEL_Q3X"|"$MODEL_Q2X"|"$MODEL_RIDGE"|"$MODEL_GSQ_RCO")
            case "$BACKEND_NAME" in
                rocm-linux|vulkan-linux|rocm-stew675-linux) DFLASH_OK=1 ;;
            esac ;;
    esac
    if [ "$DFLASH_OK" = "1" ] && [ "$BACKEND_NAME" != "rocm-stew675-linux" ]; then
        # version banner is printed to stderr
        _dver="$("$SERVER" --version 2>&1 | sed -nE 's/.*\(build ([0-9]+).*/\1/p' | head -n1)"
        [ -n "$_dver" ] && [ "$_dver" -ge 10701 ] || DFLASH_OK=0
    fi

    # GSQ-RCO ships as a separate non-MTP file (the MTP head lives only in the
    # distinct Qwen3.8-27B-GSQ-RCO-*-mtp.gguf download - confirmed via GGUF
    # tensor listing: 851 tensors, none named nextn/mtp/dflash). draft-mtp and
    # draft-mtp-adaptive need an embedded MTP head, so hide those here.
    MTP_HEAD_OK=1
    case "$MODEL" in
        "$MODEL_GSQ_RCO") MTP_HEAD_OK=0 ;;
    esac

    echo "  MTP speculative decoding:"
    _OPTS=()
    if [ "$MTP_HEAD_OK" = "1" ]; then
        _OPTS+=(normal tuned)
    fi
    _OPTS+=(off)
    if [ "$DFLASH_OK" = "1" ]; then
        _OPTS+=(dflash dflash_ngram)
    fi

    _i=1
    for _o in "${_OPTS[@]}"; do
        case "$_o" in
            normal)       printf '   [%d] Normal MTP  (draft-mtp)\n' "$_i" ;;
            tuned)        printf '   [%d] Tuned MTP   (per-backend: vk combined / stew adaptive / selfbuilt adaptive)\n' "$_i" ;;
            off)          printf '   [%d] OFF         (safe)\n' "$_i" ;;
            dflash)       printf '   [%d] DFlash2          (draft-dflash, KV4 draft, ~2x tg)\n' "$_i" ;;
            dflash_ngram) printf '   [%d] DFlash2 + ngram  (draft-dflash,ngram-mod on top)\n' "$_i" ;;
        esac
        _i=$((_i + 1))
    done
    if [ "$MTP_HEAD_OK" = "0" ]; then
        echo "   (Normal/Tuned MTP hidden: this GGUF has no embedded MTP head - use the -mtp variant for that)"
    fi
    echo

    _sel="$(pick "  MTP [1-${#_OPTS[@]}]: " "${#_OPTS[@]}")"
    case "${_OPTS[$((_sel - 1))]}" in
        normal) MTP_MODE="normal"
           # stew fork crashes MTP on qwen35moe MoE (q36 / KAT)
           if [[ "$BACKEND_NAME" == *stew* ]] && { [ "$MODEL" = "$MODEL_IQ" ] || [ "$MODEL" = "$MODEL_KATXS" ]; }; then
               SPEC_FLAG=()
               echo "  (MTP off: stew fork crashes on qwen35moe MoE)"
           elif [ "$MOE_BACKEND" = "1" ] && { [ "$MODEL" = "$MODEL_IQ" ] || [ "$MODEL" = "$MODEL_KATXS" ]; }; then
               # perf fork: cap draft depth so it fits alongside the expert cache
               if [ "$MODEL" = "$MODEL_IQ" ]; then
                   SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 3 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0)
               else
                   SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 1 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0)
               fi
           else
               SPEC_FLAG=(--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0)
               # Qwen3.8 family has one MTP layer -> n_max must be 1
               case "$MODEL" in
                   "$MODEL_Q38"|"$MODEL_Q38M"|"$MODEL_Q3X") SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                   "$MODEL_Q2X") SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
               esac
           fi ;;
        tuned) MTP_MODE="tuned"
           case "$BACKEND_NAME" in
               *stew*)
                   # stew rdna-boosts fork: adaptive spec is its signature; crashes on qwen35moe MoE
                   case "$MODEL" in
                       "$MODEL_RIDGE")
                           SPEC_FLAG=(--spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 8 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       "$MODEL_Q38"|"$MODEL_Q38M"|"$MODEL_Q3X")
                           SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       "$MODEL_Q2X")
                           SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       "$MODEL_IQ"|"$MODEL_KATXS")
                           SPEC_FLAG=()
                           echo "  (MTP off: stew fork crashes on qwen35moe MoE)" ;;
                       *)
                           # q8 dense: plain draft-mtp
                           SPEC_FLAG=(--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                   esac ;;
               *moe*)
                   # perf fork: MTP is VRAM-tight on top of the expert cache (112 slots)
                   case "$MODEL" in
                       "$MODEL_IQ")
                           SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 3 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       "$MODEL_KATXS")
                           SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 1 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       *)
                           SPEC_FLAG=(--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                   esac ;;
               *rocm-linux*|*selfbuilt*)
                   # official ROCm selfbuilt: adaptive long drafts on the q38 family (n_min3/n_max7)
                   case "$MODEL" in
                       "$MODEL_Q38"|"$MODEL_Q38M"|"$MODEL_Q3X"|"$MODEL_Q2X")
                           SPEC_FLAG=(--spec-type draft-mtp --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       "$MODEL_RIDGE")
                           SPEC_FLAG=(--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       "$MODEL_KATXS")
                           SPEC_FLAG=(--spec-type draft-mtp,ngram-mod --spec-draft-p-min 0.82 --spec-draft-n-max 2 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32) ;;
                       *)
                           # q36 MoE: plain draft-mtp
                           SPEC_FLAG=(--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                   esac ;;
               *)
                   # Vulkan backends: combined spec wins on dense Ridge/KAT
                   if [ "$MODEL" = "$MODEL_RIDGE" ] || [ "$MODEL" = "$MODEL_KATXS" ]; then
                       SPEC_FLAG=(--spec-type draft-mtp,ngram-mod --spec-draft-p-min 0.82 --spec-draft-n-max 2 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32)
                   else
                       case "$MODEL" in
                           "$MODEL_Q38"|"$MODEL_Q38M"|"$MODEL_Q3X") SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                           "$MODEL_Q2X") SPEC_FLAG=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.82 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                           *) SPEC_FLAG=(--spec-type draft-mtp --cache-type-k-draft q4_0 --cache-type-v-draft q4_0) ;;
                       esac
                   fi ;;
           esac ;;
        off) MTP_MODE="off"; SPEC_FLAG=() ;;
        dflash) MTP_MODE="off"; DFLASH_MODE="on"; SPEC_FLAG=()
           if [ -f "$DFLASH_DRAFT" ]; then
               SPEC_FLAG=(--spec-type draft-dflash -md "$DFLASH_DRAFT" --spec-draft-n-max 4 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0)
           else
               echo "  (DFlash2 draft missing: $DFLASH_DRAFT)"
               DFLASH_MODE="off"
           fi ;;
        dflash_ngram) MTP_MODE="off"; DFLASH_MODE="on"; SPEC_FLAG=()
           if [ -f "$DFLASH_DRAFT" ]; then
               SPEC_FLAG=(--spec-type draft-dflash,ngram-mod -md "$DFLASH_DRAFT" --spec-draft-n-max 4 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32)
           else
               echo "  (DFlash2 draft missing: $DFLASH_DRAFT)"
               DFLASH_MODE="off"
           fi ;;
    esac
}

# ── Per-model batch / KV config ──────────────────────────────────────────────
apply_config() {
    case "$MODEL" in
        "$MODEL_Q38"|"$MODEL_Q38M"|"$MODEL_Q3X"|"$MODEL_Q2X"|"$MODEL_Q8"|"$MODEL_GSQ_RCO")
            # dense 27B: batch >1024 collapses at 64K+ (vulkan compute buffer)
            case "$CTX_SIZE" in
                32768)  B=2048; UB=2048 ;;
                65536)  B=1024; UB=2048 ;;
                81920)  B=1024; UB=2048 ;;
                102400) B=512;  UB=512  ;;
                *)      B=512;  UB=512  ;;
            esac
            NCMOE=0 ;;
        "$MODEL_RIDGE")
            if [ "$BACKEND_IS_VK" = "1" ] && [ "$MTP_MODE" = "tuned" ]; then
                # combined-spec cfg: b1024/ub128 + q4_1 KV + cache-ram 6000
                B=1024; UB=128
                if [ "$KV_SET" != "1" ]; then
                    KV_K="q4_1"; KV_V="q4_1"
                fi
                CACHE_RAM=6000
            else
                case "$CTX_SIZE" in
                    32768) B=2048; UB=2048 ;;
                    65536|81920|102400|131072|153600) B=1024; UB=2048 ;;
                    *)     B=512;  UB=512  ;;
                esac
            fi
            NCMOE=0 ;;
        "$MODEL_KATXS")
            # KAT-MTP (MTP head adds VRAM pressure)
            if [ "$BACKEND_IS_VK" = "1" ]; then
                case "$CTX_SIZE" in
                    32768|65536) B=4096; UB=1024; NCMOE=14 ;;
                    81920)       B=4096; UB=1024; NCMOE=14 ;;
                    102400|153600) B=2048; UB=1024; NCMOE=16 ;;
                    131072)      B=2048; UB=1024; NCMOE=16 ;;
                    *)           B=512;  UB=512;  NCMOE=18 ;;
                esac
            else
                case "$CTX_SIZE" in
                    32768) B=4096; UB=1024; NCMOE=10 ;;
                    65536) B=4096; UB=1024; NCMOE=11 ;;
                    81920) B=4096; UB=1024; NCMOE=11 ;;
                    102400|153600) B=4096; UB=1024; NCMOE=13 ;;
                    131072) B=4096; UB=1024; NCMOE=13 ;;
                    *)     B=512;  UB=512;  NCMOE=17 ;;
                esac
            fi ;;
        *)  # Qwen3.6 35B-A3B MoE
            if [ "$BACKEND_IS_VK" = "1" ]; then
                case "$CTX_SIZE" in
                    32768|65536) B=4096; UB=2048; NCMOE=16 ;;
                    81920)       B=4096; UB=2048; NCMOE=16 ;;
                    102400|153600) B=4096; UB=2048; NCMOE=18 ;;
                    131072)      B=4096; UB=2048; NCMOE=18 ;;
                    *)           B=512;  UB=512;  NCMOE=21 ;;
                esac
            else
                case "$CTX_SIZE" in
                    32768|65536) B=4096; UB=4096; NCMOE=18 ;;
                    81920)       B=4096; UB=4096; NCMOE=18 ;;
                    102400|153600) B=2048; UB=2048; NCMOE=20 ;;
                    131072)      B=2048; UB=2048; NCMOE=20 ;;
                    *)           B=512;  UB=512;  NCMOE=20 ;;
                esac
            fi ;;
    esac

    # DFlash2 draft needs ~2 GiB extra compute buffers -> force small batches
    if [ "$DFLASH_MODE" = "on" ]; then
        B=512; UB=512; NCMOE=0
    fi

    # perf fork (rocm-moe): MoE expert cache - safe slot counts tuned on this GPU.
    # 112 slots leaves room for MTP draft KV: q36 MTP n_max 3 + 112 slots = 44.7 t/s,
    # KAT MTP n_max 1 + 112 slots = 61.7 t/s. 144 slots + MTP collapses (VRAM).
    MOE_CACHE_PROFILE=""; MOE_CACHE_SLOTS=0
    if [ "$MOE_BACKEND" = "1" ]; then
        case "$MODEL" in
            "$MODEL_IQ")
                [ -f "$ROOT/traces/q36-routing.csv" ] && { MOE_CACHE_PROFILE="$ROOT/traces/q36-routing.csv"; MOE_CACHE_SLOTS=112; } ;;
            "$MODEL_KATXS")
                [ -f "$ROOT/traces/kat-routing.csv" ] && { MOE_CACHE_PROFILE="$ROOT/traces/kat-routing.csv"; MOE_CACHE_SLOTS=112; } ;;
        esac
    fi

    # Linux (2026-09-04, i9-14900K): --cpu-strict 1 tested across both MoE
    # (rocm-moe-linux, Qwen3.6-35B-A3B: +21% pp, tg flat) and dense/hybrid
    # (rocm-linux, Qwen3.8-27B IQ4_XS + GSQ-RCO IQ3_S, ncmoe=0: throughput
    # neutral +-2%, but 3-4x tighter pp variance both times). No regression
    # found anywhere it was tried -> now a fixed parameter for every model/
    # backend, not conditional on MOE_BACKEND/ncmoe. Contradicts the older
    # Windows finding (neutral) - see knowledge-base/04-parameters.md "Threads".
}

# ── Launch ───────────────────────────────────────────────────────────────────
launch() {
    clear 2>/dev/null || true
    echo "================================================"
    echo "  Starting llama-server..."
    echo "================================================"
    echo "   Model     : $MODEL"
    echo "   Backend   : $SERVER"
    echo "   Context   : $CTX_SIZE   Batch: $B   UBatch: $UB   ncmoe: $NCMOE"
    echo "   KV cache  : -ctk $KV_K -ctv $KV_V   cache-ram: $CACHE_RAM"
    echo "   Sampling  : $SAMPLING_FLAG"
    echo "   Reasoning : $Q38_FLAG"
    if [ ${#SPEC_FLAG[@]} -gt 0 ]; then
        echo "   MTP/Spec  : ${SPEC_FLAG[*]}"
    else
        echo "   MTP/Spec  : (off)"
    fi
    [ "$DFLASH_MODE" = "on" ] && echo "   DFlash2   : $DFLASH_DRAFT (draft-dflash, n-max 4, KV4)"
    if [ -n "$MOE_CACHE_PROFILE" ]; then
        echo "   MoE cache : $MOE_CACHE_PROFILE ($MOE_CACHE_SLOTS slots)"
    else
        echo "   MoE cache : off"
    fi
    if [ -n "$MMPROJ_PATH" ]; then
        echo "   Vision    : $MMPROJ_PATH${MMPROJ_OFFLOAD:+ (CPU offload)}"
    else
        echo "   Vision    : off"
    fi
    echo "   Fixed     : -t 8 --cpu-strict 1 --flash-attn on --load-mode none --no-repack --fit off -np 1 --port $PORT --jinja --log-colors on -lv 3 --log-timestamps"
    echo
    echo "  Server: http://localhost:$PORT"
    echo "  Press Ctrl+C to stop"
    echo "================================================"
    echo

    [ -x "$SERVER" ] || { echo "Backend executable missing: $SERVER"; exit 1; }
    [ -f "$MODEL" ]  || { echo "Model file missing: $MODEL"; exit 1; }

    # Build the command as an array so paths containing spaces ("New Volume") stay intact.
    local cmd=("$SERVER" -m "$MODEL" -c "$CTX_SIZE" -t 8 --cpu-strict 1 -b "$B" --ubatch-size "$UB"
        -ctk "$KV_K" -ctv "$KV_V" --flash-attn on --load-mode none --n-cpu-moe "$NCMOE"
        --cache-ram "$CACHE_RAM" --no-repack --fit off -np 1 --host 0.0.0.0 --port "$PORT"
        --jinja --log-colors on -lv 3 --log-timestamps)
    # flag bundles with no spaces: intentional word-splitting.
    # SPEC_FLAG is an array (paths may contain spaces, e.g. DFlash -md draft).
    cmd+=( "${SPEC_FLAG[@]}" $SAMPLING_FLAG $Q38_FLAG )
    [ -n "$MMPROJ_PATH" ] && cmd+=(--mmproj "$MMPROJ_PATH")
    [ "$MMPROJ_OFFLOAD" = "1" ] && cmd+=(--no-mmproj-offload)
    [ -n "$MOE_CACHE_PROFILE" ] && cmd+=(--moe-cache-profile "$MOE_CACHE_PROFILE" --moe-cache-slots "$MOE_CACHE_SLOTS")

    if [ "$DRY" = "1" ]; then
        echo "  [dry-run] command:"
        printf '  '
        printf '%q ' "${cmd[@]}"
        echo
        exit 0
    fi

    # Kill stale process holding the port (fixes restart failures)
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PORT}/tcp" >/dev/null 2>&1 && sleep 1
    else
        echo "  (fuser not found - if port $PORT is busy, stop the old server first)"
    fi

    exec "${cmd[@]}"
}

# ── Main ─────────────────────────────────────────────────────────────────────
menu_model
menu_backend
menu_vision
menu_ctx
menu_kv
menu_mtp
apply_config
launch
