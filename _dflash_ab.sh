#!/usr/bin/env bash
# A/B test: Qwen3.8-27B target + DFlash2 draft (draft-dflash) vs baseline (no spec).
# Usage: _dflash_ab.sh <backend-dir> <target-model> <n-max> [port]
#   backend-dir  : rocm-linux | rocm-moe-linux | rocm-stew675-linux (under backends/bin/)
#   target-model : path to a local Qwen3.8-27B GGUF
#   n-max        : 0 = baseline (no speculative), otherwise --spec-draft-n-max N
#   port         : optional (default 8098)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRV="$ROOT/backends/bin/${1:-rocm-linux}/llama-server"
TARGET="${2:-/mnt/nvme/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf}"
DRAFT="/mnt/nvme/Users/BhatSahab/.lmstudio/models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
NMAX="${3:-7}"
PORT="${4:-8098}"
LOG="$ROOT/traces/dflash_ab_nmax${NMAX}.log"

[ -x "$SRV" ] || { echo "server not found: $SRV"; exit 1; }
[ -f "$TARGET" ] || { echo "target model not found: $TARGET"; exit 1; }
[ -f "$DRAFT" ] || { echo "draft model not found: $DRAFT"; exit 1; }

ARGS=(-m "$TARGET" -c 32768 -t 8 -b 512 --ubatch-size 512 \
      -ctk q4_0 -ctv q4_0 --flash-attn on --load-mode none \
      --n-cpu-moe 0 --cache-ram 0 --no-repack --fit off -np 1 \
      --host 127.0.0.1 --port "$PORT" --jinja -lv 4 \
      --reasoning on --reasoning-preserve)
if [ "$NMAX" != "0" ]; then
    ARGS+=(-md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max "$NMAX")
fi

echo "=== backend=$(basename "$(dirname "$SRV")") | target=$(basename "$TARGET") | n-max=$NMAX ==="
nohup "$SRV" "${ARGS[@]}" >"$LOG" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 120); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then ok=1; break; fi
done
[ "$ok" = "1" ] || { echo "SERVER FAILED TO START"; tail -30 "$LOG"; kill "$SRV_PID" 2>/dev/null; exit 1; }

grep -iE "speculative|dflash|block_size|n_max" "$LOG" | head -8

echo "--- completion (192 tokens) ---"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Solve: Janet buys 12 pens. Each pen costs $3.50. She pays with a $50 bill. How much change does she receive? Show your reasoning step by step."}],"max_tokens":192,"temperature":0.7}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("timings",{}); print("predicted_per_second: %.2f t/s (predicted_n=%d, prompt=%d tok, prompt_per_second=%.1f)" % (t.get("predicted_per_second",0), t.get("predicted_n",0), t.get("prompt_n",0), t.get("prompt_per_second",0)))' 2>/dev/null \
    || { echo "(completion parse failed)"; tail -30 "$LOG"; }

kill "$SRV_PID" 2>/dev/null
sleep 1
echo "done"
