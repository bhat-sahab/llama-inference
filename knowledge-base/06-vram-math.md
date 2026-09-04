# 06 — VRAM Math (validated)

## The budget

```
GPU total = weights(ncmoe) + KV(ctx) + RS(DeltaNet) + compute(ubatch) ≤ 16 GB
```

All numbers measured from `-lv 4` load logs on Qwen3.6-35B.

## Weights (21.27 GB file)

| ncmoe | CPU (MiB) | GPU (MiB) |
|:-----:|:---------:|:---------:|
| 20 | 9,795 | 11,989 |
| 24 | ~11,759 | ~10,025 |
| 27 | 13,043 | 8,237 |

- Each CPU layer ≈ **483-490 MiB** (up+gate+down exps)
- Formula: `ncmoe × ~490 MiB` offloaded

## KV cache (10 attention layers only — DeltaNet)

Measured: **1,440 MiB @ 262,144 ctx, q4_0** (K=720, V=720)
Formula: `10 layers × 2 × 512 (KV dim) × ctx × 0.5625 B (q4_0)`

| ctx | KV (q4_0) |
|:---:|:---------:|
| 32K | 180 MiB |
| 128K | 720 MiB |
| 256K | 1,440 MiB |

## RS buffer (recurrent state, DeltaNet)

**Fixed 62.81 MiB** — not context-dependent (R f32 2.8 + S f32 60).

## Compute buffers (ubatch-dependent)

| ubatch | compute MiB |
|:------:|:-----------:|
| 256 | 829 |
| 2048 | ~1,044 (b=2048) |
| 4096 | ~4,200 |

**This is why 256K needs b=512:** weights(10,025 n24) + KV(1,440) + RS(63) + compute(b4096: 4,200) = 15,928 → **over 16 GB → hard crash**.
With b=512 (compute ~1,400): 12,928 → fits.

## Worked examples

### 128K, Vulkan, ncmoe=18 (VALIDATED — 2 GB headroom)
```
weights 12,413 + KV 720 + RS 63 + compute 1,044 = 14,240 MiB → ✅ fits (2,064 free)
```
(why ncmoe=18 works at 128K but the same ncmoe fails at 256K)

### 256K, ncmoe=27, b=256 (VALIDATED)
```
weights 8,237 + KV 1,440 + RS 63 + compute 829 = 10,569 MiB → ✅ fits
```

### 256K, ncmoe=23, b=4096 (crashes)
```
weights 10,169 + KV 1,440 + RS 63 + compute 4,247 = 15,919 MiB → ❌ border crash
```

## Rules derived

1. **ncmoe is the only knob that moves weights off GPU** → use it to make room for KV growth
2. **ubatch sets compute-buffer size** → dropping ubatch can free 1-3 GB
3. **Fit point drifts ±2 ncmoe between reboots** (driver/memory-layout state) → keep 1 ncmoe margin on ROCm
4. One ncmoe below fit = spill = 13× pp collapse (not graceful degradation)
5. MTP heads add ~400 MB (F16) → need ncmoe+2 or ubatch-1

## KV offload toggle (`-nkvo 1`)

Moves the 1.44 GB KV to RAM → frees VRAM for weights. Tested: not needed if ncmoe tuned correctly; adds PCIe traffic on attention layers.

---

## DENSE/HYBRID models (Qwen3.8-27B) — KV comes from ATTENTION LAYERS ONLY

KV per token = `2 × n_attn_layers × kv_dim × 0.5625 B (q4_0)`.
**Qwen3.8-27B is HYBRID: 48 GDN (linear-attention, NO KV cache) + 16 full-attention layers** (3:1, verified in GGUF/config.json). KV dim = 4 kv heads × 256 = 1024.
→ `16 layers × 2 × 1024 × 0.5625 = 18,432 B = 18 KiB/token` (NOT 144 KiB — that assumed 64 attn layers).

| ctx | KV (q4_0) |
|:---:|:---------:|
| 32K | 0.56 GiB |
| 64K | 1.1 GiB |
| 128K | 2.3 GiB |
| 256K | **4.7 GiB** |

**256K FITS RAM easily** (32 GB). Earlier "dense 256K broken / 36 GiB KV" conclusions were **wrong** — those failures were GPU stuck-clocks drift artifacts. GDN layers keep a small recurrent-state buffer (~60 MiB total, fixed) instead of KV.

**Measured placement (server, `--cache-ram 0`):** Ridge @256K = PM ~17.7 GB (weights 11.7 + KV 4.7 + compute). Works: pp 720, tg 27.4 ✅

**Qwen3.6/KAT (qwen35moe):** 10 attn layers × 2 × 512 × 0.5625 = 5.76 KiB/token → 256K = 1.44 GiB (matches measured).

## Weight fit estimate (any model)

`VRAM ≈ weights − ncmoe×0.49 GB + compute(ubatch) + KV(headroom)` ≤ ~14.5 GB usable. ncmoe=0 if weights < 13 GB. One ncmoe below fit = 13× pp collapse (spill), not graceful.
