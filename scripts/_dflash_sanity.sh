#!/usr/bin/env bash
# Sanity check: greedy (temp 0) output with DFlash2 on a given backend.
# Usage: _dflash_sanity.sh <backend-dir>
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BE="${1:-rocm-linux}"
SRV="$ROOT/backends/bin/$BE/llama-server"
MODEL="/mnt/nvme/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf"
DRAFT="/mnt/nvme/Users/BhatSahab/.lmstudio/models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
PORT=8099
LOG="$ROOT/traces/dflash_sanity_$BE.log"

nohup "$SRV" -m "$MODEL" -md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max 4 \
  -c 4096 -t 8 -b 512 --ubatch-size 512 -ctk q4_0 -ctv q4_0 --flash-attn on \
  --load-mode none --n-cpu-moe 0 --cache-ram 0 --no-repack --fit off -np 1 \
  --host 127.0.0.1 --port "$PORT" --jinja --reasoning on --reasoning-preserve \
  >"$LOG" 2>&1 &
SRV_PID=$!
ok=0
for _ in $(seq 1 120); do
  sleep 2
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then ok=1; break; fi
done
[ "$ok" = "1" ] || { echo "FAILED TO START"; tail -20 "$LOG"; kill "$SRV_PID" 2>/dev/null; exit 1; }
echo "=== $BE greedy output (temp=0) ==="
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly the number: 42"}],"max_tokens":64,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=(d.get("choices") or [{}])[0]; print("content:", repr(c.get("message",{}).get("content",""))[:200]); print("reasoning:", repr(c.get("message",{}).get("reasoning_content",""))[:120]); t=d.get("timings",{}); print("tg: %.1f t/s (predicted_n=%d)" % (t.get("predicted_per_second",0), t.get("predicted_n",0)))' \
  || { echo "(parse failed)"; tail -15 "$LOG"; }
kill "$SRV_PID" 2>/dev/null
sleep 1
echo "done"
