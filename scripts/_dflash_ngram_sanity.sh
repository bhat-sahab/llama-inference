#!/usr/bin/env bash
# Sanity check: greedy (temp 0) output with DFlash2 + ngram-mod on a given backend.
# Combines draft-dflash with ngram-mod speculative decoding.
# Usage: _dflash_ngram_sanity.sh <backend-dir>
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BE="${1:-rocm-linux}"
SRV="$ROOT/backends/bin/$BE/llama-server"
MODEL="$ROOT/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf"
DRAFT="$ROOT/models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
PORT=8097
LOG="$ROOT/traces/dflash_ngram_sanity_$BE.log"

[ -x "$SRV" ]   || { echo "server not found: $SRV"; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }
[ -f "$DRAFT" ] || { echo "draft not found: $DRAFT"; exit 1; }

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
[ "$ok" = "1" ] || { echo "FAILED TO START"; grep -iE "speculativ|dflash|ngram|error|invalid" "$LOG" | tail -20; kill "$SRV_PID" 2>/dev/null; exit 1; }

echo "=== $BE greedy output (temp=0) with draft-dflash,ngram-mod ==="
grep -iE "adding speculative implementation|dflash|ngram_mod|n_match|block_size" "$LOG" | head -8
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly the number: 42"}],"max_tokens":64,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=(d.get("choices") or [{}])[0]; print("content:", repr(c.get("message",{}).get("content",""))[:200]); t=d.get("timings",{}); print("tg: %.1f t/s (predicted_n=%d)" % (t.get("predicted_per_second",0), t.get("predicted_n",0)))' \
  || { echo "(parse failed)"; tail -15 "$LOG"; }
kill "$SRV_PID" 2>/dev/null
sleep 1
echo "done"
