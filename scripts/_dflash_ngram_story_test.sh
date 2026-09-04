#!/usr/bin/env bash
# Story-generation test: ~100 word short story with draft-dflash,ngram-mod combined,
# across ALL locally available Qwen3.8-27B target quantizations, on a given backend.
# Longer, more repetitive-structure output gives ngram-mod an actual chance to draft.
# Usage: _dflash_ngram_story_test.sh <backend-dir> [start-index]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BE="${1:-rocm-linux}"
START_IDX="${2:-0}"
SRV="$ROOT/backends/bin/$BE/llama-server"
MODELS_DIR="/mnt/nvme/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF"
DRAFT="/mnt/nvme/Users/BhatSahab/.lmstudio/models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
PORT=8096

MODELS=(
  "Qwen3.8-27B-UD-Q2_K_XL.gguf"
  "Qwen3.8-27B-UD-Q3_K_XL.gguf"
  "Qwen3.8-27B-Q3_K_M.gguf"
  "Qwen3.8-27B-UD-IQ4_XS.gguf"
)

[ -x "$SRV" ]   || { echo "server not found: $SRV"; exit 1; }
[ -f "$DRAFT" ] || { echo "draft not found: $DRAFT"; exit 1; }

overall_rc=0

for idx in "${!MODELS[@]}"; do
  if [ "$idx" -lt "$START_IDX" ]; then
    continue
  fi
  m="${MODELS[$idx]}"
  MODEL="$MODELS_DIR/$m"
  LOG="$ROOT/traces/dflash_ngram_story_${BE}_${m%.gguf}.log"
  echo "=================================================================="
  echo "=== backend=$BE | target=$m | spec-type=draft-dflash,ngram-mod | 100-word story ==="
  echo "=================================================================="

  if [ ! -f "$MODEL" ]; then
    echo "SKIP: model not found: $MODEL"
    overall_rc=1
    continue
  fi

  nohup "$SRV" -m "$MODEL" -md "$DRAFT" \
    --spec-type draft-dflash,ngram-mod --spec-draft-n-max 4 \
    --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
    --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32 \
    -c 4096 -t 8 -b 512 --ubatch-size 512 -ctk q4_0 -ctv q4_0 --flash-attn on \
    --load-mode none --n-cpu-moe 0 --cache-ram 0 --no-repack --fit off -np 1 \
    --host 127.0.0.1 --port "$PORT" --jinja --reasoning on --reasoning-preserve -lv 5 \
    >"$LOG" 2>&1 &
  SRV_PID=$!

  ok=0
  for _ in $(seq 1 150); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then ok=1; break; fi
  done

  if [ "$ok" != "1" ]; then
    echo "FAILED TO START"
    grep -iE "speculativ|dflash|ngram|error|invalid" "$LOG" | tail -20
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    overall_rc=1
    continue
  fi

  echo "--- story completion (target ~100 words, max_tokens=1200, temp=0, reasoning off) ---"
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a short story of exactly about 100 words about a lighthouse keeper who discovers a message in a bottle. Plain prose only, no preamble."}],"max_tokens":1200,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import json,sys
d=json.load(sys.stdin)
c=(d.get("choices") or [{}])[0]
msg = c.get("message",{})
content = msg.get("content","") or ""
reasoning = msg.get("reasoning_content","") or ""
print("finish_reason:", c.get("finish_reason"))
if reasoning:
    print("reasoning_content (truncated 300 chars):", repr(reasoning[:300]))
print("content:")
print(content)
print()
print("word_count:", len(content.split()))
t=d.get("timings",{})
print("tg: %.1f t/s (predicted_n=%d)" % (t.get("predicted_per_second",0), t.get("predicted_n",0)))' \
    || { echo "(parse failed)"; tail -20 "$LOG"; overall_rc=1; }

  echo "--- speculative stats ---"
  grep -iE "draft acceptance|statistics.*ngram-mod|statistics.*dflash" "$LOG" | tail -5

  kill "$SRV_PID" 2>/dev/null
  wait "$SRV_PID" 2>/dev/null
  sleep 1
  echo
done

echo "=================================================================="
if [ "$overall_rc" = "0" ]; then
  echo "ALL MODELS PASSED"
else
  echo "ONE OR MORE MODELS FAILED (see above)"
fi
exit "$overall_rc"
