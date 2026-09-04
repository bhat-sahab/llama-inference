# 04 — Parameters Reference (all tested)

## The Big Three (where 90% of performance lives)

### `--n-cpu-moe N` — the VRAM fit dial (MOST IMPORTANT)
Moves the expert tensors (`blk.{0..N-1}.ffn_(up|down|gate|gate_up)_exps`) of the **first N layers** to CPU RAM. Everything else stays on GPU.

| ncmoe | Result |
|:-----:|--------|
| below fit | **13× pp collapse** (e.g. Qwen ROCm: ncmoe=19 → 236 pp vs ncmoe=20 → 3013) |
| **at fit** | max pp + max tg in working range |
| above fit | TG degrades ~1-1.5 t/s per extra layer (CPU bottleneck) |

- Sweet spot is **backend-specific**: Vulkan ncmoe=18, ROCm ncmoe=20 (Qwen 35B, 32K)
- Scales with KV growth: 128K +2, 256K +5 (Vulkan: 18→20→23)
- ncmoe=0 for models that fit entirely (Qwen3.8 dense, KAT IQ2-class)
- **MTP needs ncmoe+2** (draft heads consume VRAM)
- ⚠️ Fit point **drifts ±2 layers between reboots** on ROCm — Vulkan is stable

### `--ubatch-size N` — the pp lever
Prompt processing is chunked in ubatch-sized pieces. **Ubatch is what drives pp, not batch.**

| Qwen 35B Vulkan 32K (ncmoe=18) | pp t/s |
|:-------------------------------:|:------:|
| ub=512 | 1396 |
| ub=2048 | **2214** 🏆 |
| ub=4096 | 2175 |

- 512 → 2048 = **+58% pp**; TG flat
- Bigger ubatch = bigger compute buffers = more VRAM (can force ncmoe up)
- MTP needs ub≥2048 for efficient draft verification (b=512: tg 62→45)

### `--batch-size N` — mostly irrelevant
- b≥ub with same ub → same pp (2048/2048 vs 4096/2048 identical)
- ⚠️ Quirks by model: some collapse at `-b 4096 -ub 2048` (Bonsai: 917→341); **dense models collapse at b≥2048 when ctx ≥64K** (Q38: 628→44). IQ3_XXS collapses at b512 OR ub2048 @128K (only 1024/1024 works). **Map bad zones per model.**

## Load mode (`--load-mode` / `-lm`) — full ranking

| Mode | Qwen 21 GB pp | Bonsai pp | Notes |
|------|:------------:|:---------:|-------|
| **none** | **3013** 🏆 | **952** 🏆 | read once into RAM |
| mlock | — | 936 | locks pages, no lazy fault |
| mmap+mlock | — | 921 | locks but still lazy page-in |
| mmap | 1188 (±232 jitter) | 920 | page faults during pp |
| dio | — | 919 | direct I/O |

**Always `none`.** mmap costs 2.7× pp on big models + huge variance.

## KV cache (`-ctk`/`-ctv`)

Supported (vulkan + rocm-selfbuilt): `f16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl` (f32/q6_K/q8_1/iq4_xs rejected). Official ROCm: see 01 for DLL fix.

| Type | pp64 | Verdict |
|:----:|:----:|---------|
| **q4_0** | **410** | fastest (+13% vs f16), enough precision |
| q5_1 | 392 | |
| q4_1 | 383 | |
| iq4_nl | 376 | |
| q5_0 | 375 | |
| q8_0 | 365 | |
| f16 | 362 | baseline |

**q4_0 both K and V.** Draft KV type (`--spec-draft-type-k/v`) irrelevant — no effect.

## Flash attention

- **`--flash-attn on` mandatory** for DeltaNet/SSM models — `off` = "failed to create context"
- Required for large contexts

## Threads

| `-t` | Effect |
|:----:|--------|
| 8 | **Best** (P-cores) — tg 38.6 vs 37.4 @24 on Qwen ROCm |
| 16 | worse (HT/E-core contention) |
| 24 (default) | worse |

- CPU pinning (`-Cr 0-7 --cpu-strict 1 --prio 2`) = **neutral** at t=8 on Windows — don't bother
- Pinning matters only at t>=16 on Windows
- **Linux update (2026-09-04): see addendum below — NOT neutral on Linux**

### Linux/ROCm addendum (2026-09-04) — `--cpu-strict` is a real win here

Re-tested on Linux (Arch, kernel 7.1.9), same CPU class (i9-14900K, 8 P-cores/16 E-cores), `rocm-moe-linux` backend, Qwen3.6-35B-A3B-UD-IQ4_XS MoE model, `-t 8 -ncmoe 18 -ctk/-ctv q4_0 -fa on`, `llama-bench -p 512 -n 128 -r 3`:

| cpu-strict | cpu-range/mask | pp512 t/s | tg128 t/s |
|:---:|:---:|:---:|:---:|
| 0 (default) | none | 1049.71 +/- 155.39 | 34.89 +/- 0.53 |
| 1 | none (OS picks 8 of 32) | 1270.49 +/- 126.67 (+21% pp) | 34.84 +/- 0.41 |
| 1 | `0xff` / `-Cr 0-7` (4 physical P-cores via SMT pairs) | 1234.41 +/- 174.64 | 35.43 +/- 0.21 (tightest variance) |

This contradicts the Windows note above. On Linux, `--cpu-strict 1` alone gives +21% pp with lower variance and no tg cost. Adding an explicit `--cpu-range`/`--cpu-mask` on top gives no further pp gain (within noise) but tightens tg variance a bit more. Likely cause: Linux's CFS scheduler migrates threads across the P/E-core boundary more readily than Windows does for this workload, so `--cpu-strict` has more to fix.

**Dense/hybrid models (ncmoe=0): neutral throughput, but still tighter variance.** Same machine, `rocm-linux` backend, `-t 8 -b 2048 -ub 2048 -ctk/-ctv q4_0 -fa on`, `llama-bench -p 512 -n 128 -r 3`:

| Model | cpu-strict | pp512 t/s | tg128 t/s |
|:---|:---:|:---:|:---:|
| Qwen3.8-27B IQ4_XS | 0 | 1228.54 +/- 64.21 | 29.62 +/- 0.03 |
| Qwen3.8-27B IQ4_XS | 1 | 1250.49 +/- 15.77 (4x tighter) | 29.58 +/- 0.02 |
| Qwen3.8-27B-GSQ-RCO IQ3_S | 0 | 1063.97 +/- 36.08 | 28.42 +/- 0.01 |
| Qwen3.8-27B-GSQ-RCO IQ3_S | 1 | 1068.28 +/- 13.46 (2.7x tighter) | 28.29 +/- 0.05 |

No throughput win here (dense models have almost no CPU-side compute for thread placement to affect), but no regression either — just markedly more consistent pp numbers. Given zero downside anywhere it was tried, `--cpu-strict 1` was promoted from a conditional (MoE/ncmoe>0 only) to an **unconditional fixed parameter** in `run_model.sh` for every model and backend.

**Rule (Linux): `--cpu-strict 1` always on.** Don't hardcode a specific `--cpu-range` in shared scripts — P-core logical-ID layout is topology/BIOS-specific, while `--cpu-strict 1` alone is portable and captures nearly all the gain.

## `-ngl` (GPU layers)

- **Default (-1 = all) beats `-ngl 99`**: 3013 vs 2861 pp on ROCm (+5%). Use default, omit the flag.

## Speculative decoding (MTP)

| Flag | Finding |
|------|---------|
| `--spec-type draft-mtp` | **+38% tg** when it fits (44.6→61.6 Qwen 32K) |
| `--spec-draft-n-max` | **3 = trained default = huge spike** (61.6). 2/4/5 = 15-23 t/s disaster |
| `--spec-draft-model file` | separate draft; heads-only GGUFs need merging (see 07) |
| `--spec-draft-ngl/-n-cpu-moe` | draft placement knobs |
| iGPU draft (`--spec-draft-device dev1`) | possible only for separate-model; UHD 770 too weak |

## Server-only flags

| Flag | Effect |
|------|--------|
| `--cache-ram 0` | **REQUIRED for DeltaNet** — KV spill corrupts recurrent state |
| `--no-repack` | faster startup, 0 perf effect |
| `--fit off` | stop auto-fit (would fight ncmoe) |
| `-np 1` | single slot |
| `-nkvo 1` | move KV to RAM (frees ~1.4 GB @256K; not needed) |
| `--jinja` | use model's own chat template |
| `--spec-type none` | default; MTP off |

## Env vars

| Var | Effect |
|-----|--------|
| `GGML_SCHED_PREFETCH_EXPERTS=1/6` | prefetches CPU experts → **overflows exact-fit VRAM → 3-13× pp loss** at optimal ncmoe |
| `GGML_CUDA_REGISTER_HOST=1` | registers host memory; no benefit, adds risk |

Only helps at over-offloaded ncmoe (26+) — never at the optimal fit point. **Skip both.**

## New flags (2026-08-20 session, b10470+ / stew fork)

| Flag | Effect | Measured |
|------|--------|----------|
| `--fit-target N` | KV fit target % (auto VRAM/RAM split) | part of Ridge combined-spec config |
| `--ctx-checkpoints N` / `--swa-checkpoints` | checkpoint freq for context save/restore | part of Ridge spec config (96) |
| `--spec-type draft-mtp,ngram-mod` | **combined** MTP + ngram spec decoding | Ridge vulkan: 31.5 → **39.6** (+30%) |
| `--spec-draft-p-min P` | min draft acceptance prob (0.82 in cfg) | — |
| `--spec-draft-n-max N` | draft length (MTP trained depth = 3; 2 for combined) | 3 = trained spike |
| `--spec-ngram-mod-n-match/min/max` | ngram lookup length / bounds | 24/8/32 in cfg |
| `-ctkd/-ctvd` (`--cache-type-k-draft`) | draft KV type | q4_0 CRASHES KAT draft graph; use default |
| `--spec-type draft-mtp-adaptive` | **stew fork only** — self-tuning draft depth | Ridge: 28.7 → **44.4** (+55%) |
| `--spec-draft-n-min-adaptive N` | adaptive MTP floor (default 3) | — |
| `--no-warmup` | skip empty warmup run | part of Ridge spec cfg |
| `--reasoning-budget N` | thinking token budget | 14000 in eaman cfg |
| `--fit-target` + `--cache-ram N` | RAM KV buffer MiB | 6000 in eaman cfg |

## New env vars (2026-08-20)

| Var | Effect |
|-----|--------|
| `GGML_HIP_GDN_LANES=8` | GDN clustered-columns lanes (RDNA4, S_v=128). **8 = default in selfbuilt** (+9% pp MoE) |
| `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking":true,"reasoning_effort":"medium"}` | batch-safe kwargs JSON |

Removed/rejected: `GGML_CUDA_GDN_PRECOMPUTE_GATE` (neutral), `LLAMA_FUSED_GDN_CH_DISABLE` (crashes — no SOLVE_TRI HIP kernel).

## Batch/ubatch — ROCm vs Vulkan (Qwen 35B, b fixed)

| ubatch | Vulkan pp | ROCm pp |
|:------:|:---------:|:-------:|
| 512 | 1297 | 1346 |
| 1024 | — | **2203** (stable sweet spot, older builds) |
| 2048 | **2843** | 2696-3062 |
| 4096 | 2347 | **2861** (best) |
