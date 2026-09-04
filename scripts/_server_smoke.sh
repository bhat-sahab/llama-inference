#!/usr/bin/env bash
# Quick end-to-end smoke test: boot llama-server (q8 model) with the same
# flags run_model.sh uses, poll /health, run one tiny completion, then stop.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRV="$ROOT/backends/bin/vulkan-linux/llama-server"
MODEL="/mnt/nvme/Users/BhatSahab/.lmstudio/models/empero-ai/Qwen3.8-9B-Distill-GGUF/Qwen3.8-9B-Q8_0.gguf"
LOG="$ROOT/traces/llama-server-linux-smoke.log"
PORT=8091

nohup "$SRV" -m "$MODEL" -c 32768 -t 8 -b 2048 --ubatch-size 2048 \
  -ctk q8_0 -ctv q8_0 --flash-attn on --load-mode none \
  --n-cpu-moe 0 --cache-ram 0 --no-repack --fit off -np 1 \
  --host 127.0.0.1 --port "$PORT" --jinja --log-colors on -lv 2 --log-timestamps \
  --reasoning on --reasoning-preserve --reasoning-effort medium >"$LOG" 2>&1 &
SRV_PID=$!
echo "server pid: $SRV_PID (log: $LOG)"

ok=0
for _ in $(seq 1 90); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
        ok=1
        break
    fi
done

if [ "$ok" = "1" ]; then
    echo "HEALTH OK:"
    curl -s "http://127.0.0.1:$PORT/health"
    echo
    echo "--- completion ---"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Reply with exactly: hello from linux"}],"max_tokens":48,"temperature":0}'
    echo
else
    echo "SERVER DID NOT BECOME READY - last log lines:"
    tail -25 "$LOG"
fi

kill "$SRV_PID" 2>/dev/null
sleep 1
echo "done"
