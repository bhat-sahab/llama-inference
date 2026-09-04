# 01 — System & Builds

## Hardware

| Component | Spec |
|-----------|------|
| GPU | AMD Radeon RX 9070 XT — 16 GB VRAM (15.92 GB usable), gfx1201, 2460 MHz, 1259 MHz mem, 256-bit |
| Driver | **Adrenalin 26.7.1** (win32 32.0.31035.1003, 2026-07-24) — **validated; latest**; all 2026-08-20 results on it. Stuck-at-idle-clocks bug (tg/pp collapse) fixed by PnP reset (`tools/reset-gpu.ps1`), no reboot |
| CPU | Intel i9-14900K — 8 P-cores + 16 E-cores (32 threads) |
| RAM | 32 GB |
| OS | Windows 10.0.26100 |

## Backends & binaries (`backends/bin/`) — 2026-08-20

| Dir | Build | Notes |
|-----|-------|-------|
| `vulkan/` | **b10612** (official) | Dense/hybrid + reasoning flags. Updated 2026-08-22 (was b10488). MoE tg regressed here (see below). Launcher `Q38_DIR`. |
| `vulkan-b10181/` | **b10181** | 🏆 **MoE tg king** (45-55 t/s). Launcher `VULKAN_DIR` → MoE models. Re-downloaded after b10488 regression. |
| `vulkan-patched/` | **b10488 + pi patches** | GDN-lanes + runtime-tuning envs (`GGML_VK_GDN_LANES`, `GGML_VK_GFLOPS_PER_SUBMIT`, `GGML_VK_FA_OCCUPANCY_KIB`). **MoE tg 39.5 (+20% vs stock b10488's 32.9)**; all env sweeps neutral → keep build, ignore envs. Build: `scripts/build_vulkan_patched.ps1`. |
| `rocm/` | **b10612** (official) + **ROCm 7.14 runtime** | Updated 2026-08-22 (was b10488+7.2). 9B: pp 4274 / tg 58.6. Needs runtime DLLs (see below). |
| `rocm-72/` | **selfbuilt (patches) + ROCm 7.2** | Qwen3.6 pp **3013** 🏆 (best pp). Build: `scripts/build_rocm72.ps1` from `C:/PROGRA~1/AMD/ROCm/7.2`. |
| `rocm-stew675/` | **stew675 rdna-boosts `7a845b7`** (Aug 19) | Fork with rdna-boosts + **adaptive MTP** (`draft-mtp-adaptive`, +55% tg Ridge). Built via `scripts/build_stew675.ps1` from `llama.cpp-stew675/`. |
| `rocm-selfbuilt/` | **b10488 + custom patches + ROCm 7.14** | User build, gfx1201. Patches: **GDN fix (#27327)**, **GDN clustered-columns (LANES=8 default)**, **mmvq dynamic-warps**. Runs i-quants. MTP 48.6 (Qwen3.6). `scripts/build_official.ps1` full build, `tools/ninja_rocm.ps1` incremental. |
| `rocm-perf/` | **Fable fork `perf` (prefetch+register) + 7.14** | `GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1` → **pp +52% @ high ncmoe** (1631 vs 1074 @n26); tg = selfbuilt level. Build: `scripts/build_perf.ps1` from `llama.cpp-perf/`. |
| `rocm-moe/` | **Fable `moe-cache-overlap` branch** | MoE expert cache (`--moe-cache-*`) = **dead end on Windows/HIP** (flat all slots). Keep only for CUDA experiments. |
| `vulkan-maple/` | deepgrove fork | maple-arch only |

**⚠️ ROCm official zips ship WITHOUT runtime DLLs** → silent CPU fallback. Fix — copy runtime DLLs into the build dir:

- `rocm/` uses **ROCm 7.2 runtime** from `C:\Program Files\AMD\ROCm\7.2\bin` (hipblas/hipblaslt/rocblas/rocsolver + lib dirs; **NOT amdhip64_7.dll** — driver's `C:\Windows\System32` copy must win)
- `rocm-selfbuilt/` / `rocm-stew675/` use **ROCm 7.14** from the pip SDK

### ROCm 7.2 vs 7.14 (same patches, Qwen3.6 n1000 @32K)

| Runtime | pp2048 | tg200 |
|---------|:------:|:-----:|
| 7.14 | ~2900 | 35.3 |
| **7.2** | **3013** | **39.3** (+11% MoE tg) |

7.2 wins MoE tg (+11%); 7.14 wins Ridge slightly (+4%). Ridge: 7.2 903/28.4 vs 7.14 selfbuilt 923/26.1 (mixed).

SDK lives under **Python 3.11** (`C:\Users\BhatSahab\AppData\Local\Programs\Python\Python311\Lib\site-packages\_rocm_sdk_*`) — `python` resolves to 3.11. ROCm 7.2 installed at `C:\Program Files\AMD\ROCm\7.2`.

## Build notes (critical)

- **`tools/ninja_rocm.ps1`** = incremental rebuild of `llama.cpp-src/build_official` WITH ROCm env (vcvars + HIP_PATH/ROCM_PATH/CMAKE_PREFIX_PATH). **Bare `ninja` fails on `.cu` files** — missing env → Windows CRT `math.h` leaks in (`ldexpf`/`fabsf` host-call errors).
- `scripts/build_official.ps1` — full clean build + copy to `backends/bin/rocm-selfbuilt` + DLL staging.
- Source trees: `llama.cpp-src/` (selfbuilt, b10488 + patches, incl. uncommitted GDN work), `llama.cpp-stew675/` (stew fork, rdna-boosts branch).
- Applied patches kept in `patches/`: `rx9070xt-gdn-clustered-columns.patch`, `rx9070xt-mmvq-dynamic-warps.patch`, `rx9070xt-rocm-qwen35-concurrency.patch`, `rx9070xt-vulkan-gdn-lanes.patch`, `rx9070xt-vulkan-runtime-tuning.patch`.
- **Windows has NO `rocm-smi`** (AMD ships `hipInfo` instead) — use `tools/adlx-profile.exe` (AMD-CLI v0.1.0, Almito420/AMD-CLI, live ADLX via `amdadlx64.dll`) for power: `apply --power-limit -30|0|10` / `reset`. Verified: -30% = pp -13.6%/tg -8% (quiet); +10% = no gain (not power-limited). Launcher [8].

## GDN work on rocm-selfbuilt (session 2026-08-20)

1. **GDN fused-op fix (#27327)** — `resolve_fused_ops` device-type-aware: fused GDN stays enabled when layer-0 is CPU-pinned (was being disabled → SSM on CPU). +2-6% tg MoE.
2. **GDN clustered-columns** — `lanes_per_column` template (S_v=128, RDNA4): 2-4 columns/wave. **`GGML_HIP_GDN_LANES=8` baked as default** (was 32). Correctness 36/36 backend-ops. **+9% pp MoE (Qwen3.6), +3% pp Qwen3.8 hybrid**; tg flat.
3. **mmvq dynamic-warps** — work-aware nwarps cap for MoE matvec. Neutral (kept, harmless).
4. Rejected: gate-precompute (neutral), pp-dispatch (crashes — no `SOLVE_TRI` HIP kernel), pi-rdna-kernels (neutral; HIP `__byte_perm` is software-emulated, Q3_K/Q4_K don't use it, Q1_0 irrelevant).

## Version comparison (Qwen3.6-35B IQ4_XS, b=2048/ub=2048, ncmoe=16, same session)

| Build | Backend | pp2048 | tg200 |
|-------|---------|:------:|:-----:|
| b10181 | Vulkan | 2761-2529* | **47.3-40.6*** |
| b10434 | Vulkan | 2722 | 42.6 |
| b10435 | Vulkan | 2701 | 38.5 |
| b10453 | Vulkan | 2373 | 36.8-40 |
| b10338 | ROCm | 3057 | 41.7 |
| b10435 | ROCm | 3125 | 42.5 |
| b10453 | ROCm | 2823-2881 | 36.1-36.8 |
| b10488 | Vulkan | 1987 | **32.9** ❌ |

*Morning vs end-of-session (GPU state drift). **MoE tg trend: b10181 > everything newer** — every build regressed decode (b10488 worst: 32.9). Dense/hybrid unaffected.

## Backend comparison summary

| | Vulkan | ROCm |
|--|:------:|:----:|
| TG MoE | **b10181: 45-55** | 33-43 (selfbuilt+fix 42, stew 43.3) |
| TG dense/hybrid | 31-40 | 28-31 |
| PP dense/hybrid | 590-700 | **stew 1000-1034 / selfbuilt 880-900** |
| i-quants (IQ4_XS etc.) | ✅ | official ❌ crash (`MUL_MAT`); selfbuilt ✅; stew ✅ (b10486+) |
| **q5_1 KV cache** | ✅ | ❌ **broken** (missing HIP kernel → CPU fallback, pp ~65) |
| Stability | ✅ deterministic | ⚠️ fit drifts per boot; stuck-clocks after long sessions |
| Setup | driver only | ROCm SDK + DLL copy needed |

## GPU state drift — THE testing hazard

After long bench sessions GPU clocks stick at idle → **pp/tg collapse 2-10×** (e.g. KAT 67→33 t/s, pp 2770→1200). Many "broken config / patch doesn't work" conclusions were **artifacts of this drift**:
- KAT q4_1 "broken", Q3_K_M q4_1/q5_1 "VRAM cliff", stew q5_1 "broken" → all false (drift)
- **Fix:** PnP reset (`tools/reset-gpu.ps1`) before clean A/Bs. Same-state comparisons only. 5× retries at n≥1000 for stable tg.
