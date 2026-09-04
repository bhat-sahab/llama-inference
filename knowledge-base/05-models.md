# 05 — Models Tested (current lineup 2026-08-22)

Removed from disk & testing: Qwen3.6 Q4_K_XL, IQ4_NL_XL, Bonsai-27B, KAT IQ2_M, KAT IQ4_NL, HauhauCS-27B, Nail, Nemotron-30B-A3B, Qwen3.8 Q3_K_S, Qwen3.8 IQ3_XXS, maple (see git history).

**⚠️ All "Qwen3.8-27B" models are HYBRID, not dense:** 48 GDN (linear-attention/DeltaNet) + 16 full-attention layers (3:1, `layer_types` in config.json, verified in GGUF tensors). `qwen35` arch without MoE. KV cache comes from the 16 attn layers only.

## 1. Qwen3.6-35B-A3B UD-IQ4_XS MTP (18.2 GB) — MoE workhorse

- Arch: qwen35moe, 41 blocks, 256 experts / 8 active, DeltaNet hybrid (10 attn layers), **1 MTP head**
- Path: `~/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/` (+ `mmproj-F32.gguf`)
- **FULL BACKEND TABLE @32K (2026-08-22) — MTP = plain draft-mtp n_max=3 + draft KV q4_0:**

| Backend | no-MTP tg | +MTP tg | pp2048 |
|---|---|---|---|
| **vulkan b10488** | 32.9 | **51.6-56.6** 🏆 | ~1987 |
| **vulkan b10181** | **45.8** | 53.2 | ~2761 |
| rocm-selfbuilt | 36.4 | **48.6** | ~2900 |
| rocm-perf + envs | 33.8 | 48.5 | 1631 @n26 |
| rocm-official (7.2 rt) | 34.7 | 46.0 | 675* |
| rocm-72 (7.2) | 35.2 | 45.4 | 3013 (hist) |
| rocm-stew675 | 35.1 | 41.1 | ~3089 |
| vulkan-patched | 31.9 | 38.8 | ~1108* |
| rocm-moe (moe-cache) | 28.1 | — | cache flat |

*server pp (443 tok). MTP works on ALL backends incl stew (old "stew MTP crashes qwen35moe" = STALE).

## 2. Kwaipilot KAT-Coder V2.5 Dev IQ4_XS MTP (19.7 GB) — fastest coder

- Arch: qwen35moe, 256 experts / 8 active, IQ4_XS 4.25 bpw, MTP head
- Path: `~/.lmstudio/models/thread13/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF-MTP/` (bartowski.mtp build)
- **ROCm stew675: pp 2613-2850, tg 53-56** 🏆 (n10 @32K)
- Vulkan b10181: ncmoe **14** (was 9 for non-MTP) → pp 2179, tg 47.2
- MTP flag: **OFF** (KAT MTP = 35.4 vs 53.1 baseline; KAT + Ridge combined-spec = 45.2, still below no-MTP)
- Non-MTP file (bartowski 18.8 GB): tg 56.4 selfbuilt — MTP head costs nothing

## 3. Qwen3.8-27B UD-IQ4_XS (14.3 GB) — hybrid, 4-bit quality

- Arch: qwen35 hybrid, 64 layers (48 GDN + 16 attn), MTP 1
- Path: `~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/`
- Vulkan b10488 @32K: pp 589, tg 28.4 (new 13.3 GiB build fits — old 15.7 GB one CPU-fell-back)
- KV ~18 KiB/token → 256K = 4.7 GiB (fits RAM)
- **Vision ✅** — `mmproj-F16.gguf` (same dir): sees images (verified red-square test)
- **⚠️ 128K MUST be b512/512** — b1024/2048 = VRAM spill (13.3 + 2.3 + buffers > 15.9) → tg ~12; b512/512 → 30.5. 256K = KV spill (tg 11) → ❌

## 4. Qwen3.8-27B Q3_K_M (13.8 GB) — hybrid, best tg of the Q38 family

- Vulkan b10488 @32K: pp 655, tg 37.4 (q4_0 KV)
- q4_1 → 38.6, q5_1 → 39.4 (vulkan; slight gain, higher KV precision)
- 64K/128K: pp 646-670, tg 31-32. 256K: retest pending (KV fits now)
- **Vision ✅** — `mmproj-F16.gguf` (same dir, unsloth) loads + sees images
- **128K b1024/2048 = SPILL (tg 11.4); b512/512 → pp 570, tg 34.7** ✅. 256K = KV spill → ❌

## 5. Qwen3.8-27B Ridge-3.7bpw (12.6 GB) — hybrid, ALL contexts ✅

- Path: `~/.lmstudio/models/empero-ai/Qwen3.8-27B-Ridge-GGUF/`
- **Only Q38-family model that works at 256K** (pp 720, tg 27.4)
- Vulkan b10488: pp 597-885, tg 31.5; +combined spec (MTP+ngram, q4_1 KV) → **39.6** (spikes 96)
- stew675 + adaptive MTP → **44.4** 🏆; stew/selfbuilt no-spec → 28-31
- Contains junk `blk.64` tensors (~430 MB, ignored with warning)
- **Vision ✅** — `mmproj-Qwen3.8-27B-BF16.gguf` (same dir, empero): verified, tg 34.6 w/ vision

## 6. Qwen3.8-9B-Distill Q8_0 (9.1 GB) — small hybrid, huge MTP gain

- Arch: qwen35 hybrid (small), 1 MTP head
- Path: `~/.lmstudio/models/empero-ai/Qwen3.8-9B-Distill-GGUF/`
- **Q8_0 weights + q8_0 KV** (quality pick; Q4_K_M was faster but removed)
- ROCm selfbuilt @32K (q8_0 KV): pp 2958, tg **58.2 no-MTP → 83.4 with `--spec-type draft-mtp` (+43%)** 🏆
- No 9B-compatible mmproj on disk (27B ones die — hidden-dim mismatch). **Vision ❌ (no mmproj)**
- Removed: Q4_K_M (5.8 GB, pp 2626, tg 84, MTP 120.4 selfbuilt) — faster but lower quality

## Leaderboard (current lineup, best config)

| Rank | Model | Backend | tg t/s | pp t/s |
|:----:|-------|:-------:|:------:|:------:|
| 1 | Qwen3.8-9B Q8_0 + MTP *(9B, small)* | ROCm selfbuilt | **83.4** | ~2958 |
| 2 | Qwen3.6 IQ4_XS + MTP | **vulkan b10488** | **51.6-56.6** | ~1987 |
| 3 | KAT-Coder IQ4_XS MTP (n10) | ROCm stew | **~55** | ~2620 |
| 4 | Qwen3.6 IQ4_XS + MTP | vulkan b10181 | 53.2 | ~2761 |
| 5 | Qwen3.6 IQ4_XS + MTP | ROCm selfbuilt | 48.6 | ~2900 |
| 6 | Qwen3.6 IQ4_XS + MTP | rocm-perf + envs | 48.5 | 1631@n26 |
| 7 | Qwen3.6 IQ4_XS (no-MTP) | vulkan b10181 | 45.8 | ~2600 |
| 8 | Ridge + adaptive MTP | ROCm stew | 44.4 | ~872 |
| 9 | Ridge + combined spec | Vulkan b10488 | 39.6 | ~640 |
| 10 | Q3_K_M | Vulkan b10488 | 37.4 | ~655 |
| 11 | Ridge no-spec | Vulkan/ROCm | 28-31 | 600-1030 |
| 12 | Qwen3.8 IQ4_XS | Vulkan b10488 | 28.4 | ~589 |

Historical speed records (removed): KAT IQ2_M 141 t/s · Bonsai-27B Q1_0 59 t/s · Qwen3.6 Q4_K_XL +MTP 61.6 t/s.
