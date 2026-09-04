# 09 — Checklists

## ALWAYS DO

- [ ] `--load-mode none` (never mmap/mlock)
- [ ] `-ctk q4_0 -ctv q4_0`
- [ ] `--flash-attn on` (DeltaNet models fail to load without it)
- [ ] `-t 8`
- [ ] `--n-cpu-moe` at the fit point (or 0 if model fits)
- [ ] `--cache-ram 0` (DeltaNet correctness)
- [ ] `--no-repack --fit off -np 1`
- [ ] Cold pp (fresh server, first request)
- [ ] Repeat ×2 for tg stability
- [ ] Check `build:` line — assets update silently

## NEVER DO

- [ ] Don't change `--spec-draft-n-max` from 3
- [ ] Don't set GGML_SCHED_PREFETCH_EXPERTS / GGML_CUDA_REGISTER_HOST at the fit point
- [ ] Don't use `-ngl 99` (default -1 is faster)
- [ ] Don't read pp from a warm/cached request
- [ ] Don't use b>512 at 256K context (ROCm hard-crashes)
- [ ] Don't trust ROCm fit across reboots (add +1 margin)
- [ ] Don't run llama-bench for `-c` context tests (bench has no -c; use server)
- [ ] Don't write .bat files with LF endings
- [ ] Don't expect ROCm to run ternary (Q1_0/Q2_0) models fast

## Config decision tree

```
Does model fit (size < 13 GB)?
├─ YES → ncmoe=0, b4096/ub2048 (or 4096/4096 if quirk)
│        MTP if heads exist (ncmoe stays 0, ub≥2048)
└─ NO → find fit ncmoe (start 18, cliff-walk by ±2)
        b4096/ub2048, MTP needs ncmoe+2
        ctx > 128K: ncmoe+2, 256K → b512/ub512

Backend?
├─ Vulkan → default (stable, fast)
└─ ROCm  → b4096/ub4096, +1 ncmoe margin for restart drift
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| pp ~200-300 (bench) | VRAM spill — ncmoe below fit | ncmoe +2 |
| pp 40-60 on "repeat" (server) | prompt cache hit | cold test / fresh prompt |
| server dies on first request | compute buffers overflow at request time | smaller b/ub, higher ncmoe |
| `device: -1` / launch failure | VRAM exhausted during load | ncmoe +2, b=512 |
| "failed to create context" | fa off on SSM model | `--flash-attn on` |
| draft "missing tensor token_embd" | heads-only draft GGUF | merge heads into main (07 §10) |
| draft "missing rope.dimension_sections" | draft lacks arch metadata | merge all missing KV from main |
| huge ±std on pp | mmap page faults | `--load-mode none` |
| tg 15-25 with MTP on | n_max wrong / draft spills | n_max=3, ncmoe+2 |
| `.bat` "syntax of command is incorrect" | LF line endings | convert to CRLF |
