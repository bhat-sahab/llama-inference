# llama.cpp Benchmark Knowledge Base

Complete findings from intensive tuning of **Qwen3.6 / KAT-Coder / Qwen3.8** models on an **AMD RX 9070 XT (16 GB, RDNA4/gfx1201)** with **Intel i9-14900K**, Windows. Last updated 2026-08-22.

## Contents

| File | What it covers |
|------|----------------|
| [01-system.md](01-system.md) | Hardware, software, binaries, build history, GDN patches |
| [02-best-configs.md](02-best-configs.md) | **Final recommended configs** — per model, per context, per backend |
| [03-methodology.md](03-methodology.md) | Exactly how we tested (bench + server) |
| [04-parameters.md](04-parameters.md) | Every parameter, what it does, measured effect |
| [05-models.md](05-models.md) | Every model tested + full results |
| [06-vram-math.md](06-vram-math.md) | Memory budget calculations, validated (KV corrected: hybrid = 18 KiB/token) |
| [07-discoveries.md](07-discoveries.md) | Key insights & pitfalls (read this!) |
| [08-testing-new-models.md](08-testing-new-models.md) | Step-by-step guide for new models |
| [09-checklists.md](09-checklists.md) | Do/Don't checklists |
| [reddit-KAT-post.md](reddit-KAT-post.md) | KAT-Coder reddit post draft |

## TL;DR — The rules (2026-08-22 update)

1. **`--load-mode none`** always — mmap = 2.7× pp loss on big models; **`--no-mmap` = VRAM/`device -1` OOM at 64K+ with cache**
2. **`-ctk q4_0 -ctv q4_0`** — safest KV type. q4_1/q5_1: vulkan-only win (Ridge spec); **q5_1 broken on ROCm**
3. **`--flash-attn on`** — mandatory for DeltaNet/SSM models
4. **`-t 8`** — P-cores only
5. **`--n-cpu-moe N`** = VRAM fit dial — 1 layer below fit = 13× pp collapse
6. **ubatch drives pp, ncmoe drives tg** — max ubatch that fits, min ncmoe that fits
7. **Qwen3.8-27B = HYBRID (48 GDN + 16 attn), NOT dense** — KV 18 KiB/token, 256K fits (4.7 GiB); Ridge runs 256K
8. **KV for qwen35moe = 5.76 KiB/token** (10 attn layers) — 256K = 1.44 GiB
9. **i-quants (IQ4_XS…): vulkan + rocm-selfbuilt + stew** — official ROCm crashes (`MUL_MAT`)
10. **Qwen3.6 MTP KING = vulkan b10488 (52-57 t/s, plain draft-mtp n_max=3)** — beats selfbuilt 48.6; **stew MTP works now (41.1, +17% — old "crash" = stale)**; KAT MTP = OFF; Ridge adaptive (stew) +55%
11. **vulkan b10181 = no-MTP tg king** (45.8); MTP works on it too (53.2)
12. **GPU stuck-clocks drift poisons A/Bs** — PnP reset before series, 5× retries, n≥1000, trust tg over pp
13. **ROCm 7.2 > 7.14 for MoE tg** (+11%); rocm-official (7.2 rt) + MTP = 46.0
14. **Launcher auto-kills stale PID on port 8081** — a leftover server makes restarts exit code 1 ("crash")
15. **Q8_0 weights → pair with q8_0 KV** (9B: tg 58.2 → 83.4 MTP); q4_0 KV still fastest elsewhere
16. **Qwen3.8-27B family = vision-capable** (mmproj on disk); 9B has none. Thinking models answer in `reasoning_content`
17. **rocm-perf (Fable fork) + envs = pp +52% @ high ncmoe** (1631 vs 1074) — pp-only; tg = selfbuilt level
18. **Fable moe-cache = DEAD END on Windows/HIP** — flat at every slots count; REGISTER_HOST makes copies cheap already; kernels CUDA-only-tuned

## Speed leaderboard (current lineup, best config, 2026-08-22)

| Model | Backend | tg t/s | pp t/s |
|-------|:-------:|:------:|:------:|
| Qwen3.8-9B Q8_0 +MTP *(9B)* | ROCm selfbuilt | **83.4** | ~2958 |
| **Qwen3.6 IQ4_XS +MTP** | **vulkan b10488** 🏆 | **51.6-56.6** | ~1987 |
| KAT-Coder IQ4_XS MTP (n10) | ROCm stew675 | **~55** | ~2620 |
| Qwen3.6 IQ4_XS +MTP | Vulkan b10181 | 53.2 | ~2761 |
| Qwen3.6 IQ4_XS +MTP | ROCm selfbuilt | 48.6 | ~2900 |
| Qwen3.6 IQ4_XS +MTP | rocm-perf + envs | 48.5 | 1631@n26 |
| Qwen3.6 IQ4_XS (no-MTP) | Vulkan b10181 | 45.8 | ~2600 |
| Ridge + adaptive MTP | ROCm stew675 | 44.4 | ~872 |
| Ridge + combined spec | Vulkan b10488 | 39.6 | ~640 |
| Ridge + vision | Vulkan b10488 | 34.6 | — |
| Q3_K_M | Vulkan b10488 | 37.4 | ~655 |
| Ridge no-spec | Vulkan/ROCm | 28-31 | 600-1030 |
| Qwen3.8 IQ4_XS | Vulkan b10488 | 28.4 | ~589 |

Historical records (removed models): KAT IQ2_M 141 t/s · Bonsai-27B 59 t/s · Qwen3.6 Q4_K_XL +MTP 61.6 t/s

## Tools (in `../tools/`)

- `run_model.bat` (root) — interactive launcher (models × backend × context × MTP × GPU reset × ADLX power; auto-kills stale port 8081)
- `server_test.py` — spawn server → health-poll → completion → pp/tg (`--ctk/--ctv/--spec` options added)
- `vision_test.py` — spawn server + mmproj → health-poll → `/v1/chat/completions` with generated image → answer + pp/tg
- `mtp_test.py` — MTP spec-decode tests
- `gguf_info.py` — GGUF scalar metadata
- `gpu_check.ps1` / `gpu_monitor.ps1` — GPU/VRAM/clock state sampling
- `reset-gpu.ps1` — PnP cycle (unstuck idle clocks)
- `adlx-profile.exe` / `adlx-apply.exe` — AMD-CLI live power tuning (rocm-smi does NOT exist on Windows)
- `ninja_rocm.ps1` — incremental ROCm rebuild WITH env (bare ninja fails on .cu)
- Build scripts: `scripts/build_official.ps1` (selfbuilt 7.14), `scripts/build_rocm72.ps1` (ROCm 7.2), `scripts/build_stew675.ps1` (stew fork), `scripts/build_vulkan_patched.ps1`
- Applied patches: `patches/`
