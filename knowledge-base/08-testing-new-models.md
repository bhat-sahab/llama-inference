# 08 — How to Test a New Model (full playbook)

Covers **MoE + hybrid + dense + i-quant + vision + MTP**. ~30-45 min per model. Use the tools in `../tools/`.

## Step -1 — GPU state check (IMPORTANT, 2026-08-20)

**GPU stuck-clocks poison everything.** Before any series:

```powershell
# quick check with a known model (e.g. Qwen3.6 @32K n16): expect pp ~2800 / tg ~45 (vulkan-b10181)
# if you see pp ~1400 / tg ~30 → GPU is stuck → run:
powershell -File tools\reset-gpu.ps1   # PnP cycle, needs admin, screen flickers
```

Protocol for all A/Bs: PnP reset → interleave configs → **5× retries at n≥1000** → compare **tg** (stable) not pp (state-sensitive). Trust nothing from a single sample.

## Step 0 — Inspect the model

```bash
ls -lh <model.gguf>                          # size → fit estimate
python gguf_info.py <model.gguf>             # arch, block_count, expert_count, expert_used_count
```

Or for full metadata (incl. MTP heads / kv heads):

```bash
python -c "
import sys; sys.path.insert(0, 'D:/Work/llama/llama.cpp-src/gguf-py')
from gguf import GGUFReader
r = GGUFReader(r'<model.gguf>')
print('arch:', r.fields['general.architecture'].contents())
for k in r.fields:
    if any(s in k for s in ['block_count','expert','nextn','head_count','embedding_length']):
        print(' ', k, '=', r.fields[k].contents())
print('nextn tensors:', sum(1 for t in r.tensors if 'nextn' in t.name))
"
```

Answers needed: **arch** (dense/MoE/hybrid), **blocks**, **experts + active count**, **MTP heads?**, **size**.

### Classify the model — this decides the whole test path

| Type | Example | Behavior |
|------|---------|----------|
| MoE sparse | KAT IQ4_XS, Qwen3.6 IQ4_XS | needs `--n-cpu-moe` fit dial; ncmoe drives tg |
| Dense | Qwen3.8 Q3_K_S | ncmoe=0; batch collapses at 64K+; KV huge |
| i-quant | IQ3_XXS, IQ2_M | vulkan-only (ROCm official crashes `MUL_MAT`); bench may hang |
| Hybrid/DeltaNet | Qwen3.6 | needs FA on + `--cache-ram 0` |

## Step 1 — VRAM & KV math (before running anything)

```
KV/token (q4_0) = 2 × n_layers × kv_dim × 0.5625 B     # kv_dim = kv_heads × head_dim (check attn_qkv tensor dims)
```

Dense 27B ≈ 144 KiB/token (36 GiB @256K — impossible, goes to RAM). MoE with few attn layers ≈ KBs/token. See 06-vram-math.md.

Fit: `weights − ncmoe×0.49 GB + compute(ubatch) ≤ ~14.5 GB`. ncmoe=0 if weights < 13 GB.

## Step 2 — Baseline bench (find the fit point / bad zones)

```bash
# MoE — sweep ncmoe 0, then +2 steps
llama-bench -m <model> -p 2048 -n 200 -t 8 -ctk q4_0 -ctv q4_0 -fa 1 \
  -b 4096 -ub 2048 -lm none --n-cpu-moe <N>
```

- **The fit cliff is a cliff**: one ncmoe below fit = pp collapses ~13× (spike to ~60-200). That's the fit point.
- **Bad zones vary by model+quant**: known combos — Q3_K_S ub=1024 collapses; IQ3_XXS ub=2048 collapses at 128K; KAT ub=1024 fine. Map by sweeping b/ub ∈ {4096/2048, 2048/2048, 2048/1024, 1024/1024, 512/512}.
- **bench can hang** (IQ3_XXS after table header) → fall back to server tests.
- TG is flat across batches; pp is the batch lever.

## Step 3 — Server validation (real numbers, per context)

Use the helper — spawns server, polls /health, posts completion, prints pp/tg:

```bash
python server_test.py --server D:/Work/llama/backends/bin/vulkan/llama-server.exe \
  -m "<model>" -c 32768 -b <B> -ub <UB> -ncmoe <N> -p 1000 -n 200 \
  --extra "--jinja --reasoning on --reasoning-preserve" --health-timeout 90
```

Rules:
- **Cold first request only** for pp (prompt-cache artifact makes repeats look 10× slower).
- Thinking models need `--jinja` + reasoning flags; use a chat-shaped prompt or the fox-prompt generator (some models EOS after 1 token on garbage — tg unreliable then, retry with real text).
- temp ≥ 0.5 for speculative modes (temp 0.0 + MTP = 1 token).
- Monitor VRAM while running: `powershell -File gpu_check.ps1` / `gpu_monitor.ps1` (also catches GPU state drift).

### Context ladder — check 32K → 64K → 128K → 256K

Per-context rules learned:
- **Dense**: b2048 ok @32K; b1024/ub2048 @64K-128K; 256K broken (skip — attention fallback). MoE: ncmoe +2 per ctx doubling; **256K forces b512/ub512** (compute buffers).
- KV grows 4× per doubling — watch VRAM; if pp collapses to ~30-60 with GPU pegged, it's the attention-kernel/VRAM ceiling, not clocks.

## Step 4 — Backend matrix (vulkan → rocm → rocm-selfbuilt)

| Backend | When |
|---------|------|
| `vulkan/` (b10181) | MoE tg king. Default. |
| `vulkan-new/` (b10470) | dense/thinking (needs `--reasoning-effort`/`--chat-template-kwargs`) |
| `rocm/` (b10470) | good pp for MoE/dense-K; **crashes on i-quants**; needs DLL fix (see 01) |
| `rocm-selfbuilt/` | i-quants work; user build — don't touch/rebuild casually |

Dense models: vulkan ≥ everything (ROCm dense pp ~3× worse). i-quants: vulkan only.

## Step 5 — MTP (if `nextn` heads exist)

```bash
--spec-type draft-mtp        # n_max = trained default (3). ngram: --spec-type ngram-mod --spec-ngram-mod-n-max 2
```

- MoE: MTP strong (Qwen3.6 +38% tg) but needs ncmoe+2 for draft-head VRAM + ub≥2048.
- **Dense: MTP weak (+5-7%)** — same-model draft, low acceptance. Test it, expect little.
- MTP may force smaller batch at high ctx (Q3_K_S @128K: b512/512; IQ3_XXS: b1024/1024 fine).

## Step 6 — Vision (if mmproj exists)

`--mmproj <mmproj.gguf>` — add ~0.9 GB VRAM, pp -5%. Test same configs.

## Step 7 — Sampling + chat sanity

Thinking/chat models: `--reasoning on --reasoning-preserve`; effort via `--reasoning-effort medium` (b10434+) or env `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"reasoning_effort": "medium", ...}` (batch-safe). Sampling baseline: temp 0.6, top-k 20, top-p 0.95, min-p 0.05, repeat-penalty 1.05.

## Step 8 — Record & integrate

1. Add per-context table to `knowledge-base/05-models.md` + `02-best-configs.md`.
2. Add to `run_model.bat` (model path + per-context b/ub/ncmoe + flags + expected pp/tg in `:set_speeds`).
3. If you need flags the active build rejects → pick the build that supports them (see 01 flag table).

## Full checklist

- [ ] Inspected: arch / experts / MTP / size / kv dims
- [ ] Fit point mapped (ncmoe sweep, cliff noted)
- [ ] Batch sweep done (bad zones mapped per ctx)
- [ ] Server numbers at 32K/64K/128K(/256K) — cold pp only
- [ ] Backend matrix (vulkan + rocm) — i-quant crash check
- [ ] MTP tested (if heads) — accept/reject decision
- [ ] Vision tested (if mmproj)
- [ ] VRAM/RAM placement checked (KV in RAM? GPU state sane?)
- [ ] Restart stability (GPU state drift accounted for)
- [ ] Launcher + KB updated

## Pitfalls recap (the expensive ones)

- bench hangs ≠ server broken (i-quants)
- ROCm official zip missing runtime DLLs → silent CPU fallback (check `ggml_cuda_init` line in log!)
- ROCm i-quant = `MUL_MAT failed` crash
- dense b≥2048 @64K+ = pp collapse to ~45
- 256K dense = broken; 256K MoE = b512/512 only
- prompt-cache artifact on repeated requests
- GPU state drift across a session → same-build-only comparisons, or PnP reset
- `-khad`/`-vhad`/`--merge-qkv` don't exist in any current build
- `.llg`/server logs at `-lv 3` for `-t`/KV/buffer info; some builds don't print KV sizes — measure VRAM via `gpu_monitor.ps1` instead
