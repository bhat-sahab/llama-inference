#!/usr/bin/env bash
# A/B test: perf fork llama-server with MoE expert cache ON vs OFF + optional spec flags.
# Usage: _moe_ab.sh <on|off> <slots> [model] [profile] [extra-server-args...]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRV="$ROOT/backends/bin/rocm-moe-linux/llama-server"
MODEL="${3:-/mnt/nvme/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
PROFILE="${4:-$ROOT/traces/q36-routing.csv}"
PORT=8097

CACHE="${1:-on}"   # on | off
SLOTS="${2:-144}"

run_server() {  # run_server <logfile>
    local log="$1"
    local args=(-m "$MODEL" -ncmoe 18 -fa on -c 8192 --load-mode none -t 8 --port "$PORT" -lv 4)
    if [ "$CACHE" = "on" ]; then
        args+=(--moe-cache-profile "$PROFILE" --moe-cache-slots "$SLOTS")
    fi
    shift
    while [ $# -gt 0 ]; do args+=("$1"); shift; done
    nohup "$SRV" "${args[@]}" >"$log" 2>&1 &
    SRV_PID=$!
    local ok=0
    for _ in $(seq 1 60); do
        sleep 2
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then ok=1; break; fi
    done
    [ "$ok" = "1" ] || { echo "SERVER FAILED TO START"; tail -20 "$log"; exit 1; }
}

echo "=== CACHE $CACHE (slots=$SLOTS) | model=$(basename "$MODEL") | spec: ${*:5} ==="
run_server "traces/moe_ab_${CACHE}_${SLOTS}.log" "${@:5}"
grep -i 'init_moe_expert_cache\|pack allocation' "traces/moe_ab_${CACHE}_${SLOTS}.log" | head -1
echo "--- completion (96 tokens) ---"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a detailed explanation of how Mixture of Experts works, with a concrete example."}],"max_tokens":96,"temperature":0}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("timings",{}); print("predicted_per_second: %.2f t/s (predicted_n=%d, prompt=%d tok, prompt_per_second=%.1f)" % (t.get("predicted_per_second",0), t.get("predicted_n",0), t.get("prompt_n",0), t.get("prompt_per_second",0)))' 2>/dev/null \
    || echo "(completion parse failed)"
kill "$SRV_PID" 2>/dev/null
sleep 1
echo "done"
