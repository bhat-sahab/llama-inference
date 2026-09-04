# 03 — Testing Methodology

Two measurement tools, two different numbers. **Know which is which.**

## 1. llama-bench (synthetic, warm, peak capability)

```
llama-bench.exe -m <model> -p 2048 -n 200 -t 8 -ctk q4_0 -ctv q4_0 \
  -fa 1 -b 4096 -ub 2048 -lm none --n-cpu-moe 20 -r 3
```

- Warms up (first rep discarded), repeats, reports mean ± std
- **pp2048 / tg200** — the trustworthy peak numbers
- Does NOT take `-c` (context) — KV sized to prompt
- No `--no-repack`/`--cache-ram`/`--fit` — server-only flags
- Any numeric param accepts lists/ranges: `-p 512,1024,2048` or `-p 512-4096+2`

## 2. llama-server (realistic, cold first request)

```
llama-server.exe -m <model> -c <ctx> -t 8 -b <b> --ubatch-size <ub> \
  -ctk q4_0 -ctv q4_0 --flash-attn on --load-mode none \
  --n-cpu-moe <n> --cache-ram 0 --no-repack --fit off -np 1 \
  --host 127.0.0.1 --port 8099

# wait for health (poll /health until 200), then:
curl -s -X POST http://127.0.0.1:8099/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"...~1000 tokens...","n_predict":100,"temperature":0}'
```

Parse `timings` from response:
- `prompt_n / (prompt_ms/1000)` = pp t/s
- `predicted_n / (predicted_ms/1000)` = tg t/s

## ⚠️ Critical artifacts

### Prompt-cache artifact (the "pp collapse" mirage)
The server caches the prompt in KV across requests. A SECOND request with the **same prompt** only evaluates the changed tail (~4-9 tokens):
```
"cache_n": 1037,  "prompt_n": 9    ← 9 tokens evaluated, pp looks like 40 t/s!
```
**Not a slowdown — it's the cache working.** Always test pp on a fresh server (first request) OR use different prompts per request.

### Cold vs warm
- First request after load: graph build + kernel JIT included
- Subsequent: cached path
- For tg (steady-state) both agree; for pp only trust the cold first request

### Windows background processes
Each terminal shell kills its background children when the shell exits. Run server + test in **one command**, or the server dies silently.

## Standard test matrix for a config

| Test | Purpose |
|------|---------|
| bench `-p 2048 -n 200 -r 3` | Peak pp/tg |
| bench `-p 1024 -n 0 -r 3` | pp sweep speed (fast) |
| server + 1K prompt | Real-world numbers |
| Repeat ×2 on same server | Stability check |
| Restart + retest | Fit-point stability (ROCm) |

## Automation

`../ctx_test.py` runs the full server-based context sweep (32K/64K/128K/256K) with per-backend configs:

```
python ctx_test.py vulkan
python ctx_test.py rocm-selfbuilt
```
