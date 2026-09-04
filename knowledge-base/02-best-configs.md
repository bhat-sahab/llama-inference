# 02 — Best Configs (Validated, 2026-08-20)

## Common flags (ALL models, ALL contexts)

```
-t 8 -ctk q4_0 -ctv q4_0 --flash-attn on --load-mode none
--cache-ram 0 --no-repack --fit off -np 1 --host 0.0.0.0 --port 8081 --jinja
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

`--load-mode none` always (mmap = 2.7× pp loss). **KV q4_0 everywhere** (q4_1/q5_1 only help vulkan Ridge-spec; q5_1 BROKEN on ROCm). `--cache-ram 0` = KV spills to RAM when VRAM is full.

---

## 1. Qwen3.6-35B-A3B UD-IQ4_XS **MTP** (18.2 GB, MoE 256/8 + 1 MTP head) — Vulkan `vulkan-b10181/`

| Context | `-b`/`ub` | ncmoe | pp t/s | tg t/s |
|:-------:|:---------:|:-----:|:------:|:------:|
| 32K | 4096/2048 | 16 | ~2600 | ~46 |
| 64K | 4096/2048 | 16 | ~2500 | ~45 |
| 128K | 4096/2048 | 18 | ~2400 | ~44 |
| 256K | 512/512 | 21 | ~1200 | ~42 |

ROCm (rocm-selfbuilt, ncmoe 18/18/20/20, b4096/ub4096→b512/512): pp ~2800-2900, tg **35-37 baseline → 48.6 with `--spec-type draft-mtp` (+31%)**. 🏆 **vulkan MTP beats it** — b10488 + plain `draft-mtp` n_max=3 → **51.6-56.6**; b10181 + MTP → **53.2** (2026-08-22). MTP works on vulkan b10488/b10181 + selfbuilt + stew (2026-08-22 — old "stew MTP crashes qwen35moe" = STALE).

### Full backend table @32K (2026-08-22, MTP = plain draft-mtp n_max=3 + draft KV q4_0)

| Backend | no-MTP tg | +MTP tg | Δ |
|---|---|---|---|
| **vulkan b10488** | 32.9 | **51.6-56.6** 🏆 | +57-72% |
| **vulkan b10181** | 45.8 (hist) | **53.2** | +16% |
| rocm-selfbuilt | 36.4 | **48.6** | +31% |
| rocm-perf + envs | 33.8 | **48.5** | +44% |
| rocm-official (7.2 rt) | 34.7 | **46.0** | +33% |
| rocm-72 (7.2) | 35.2 | **45.4** | +29% |
| rocm-stew675 | 35.1 | 41.1 | +17% |
| vulkan-patched | 31.9 (chat) | 38.8 | +22% |

MTP ranking: **vulkan b10488 (52-57) > b10181 (53) > selfbuilt (48.6) ≈ rocm-perf (48.5) > rocm-official (46.0) > rocm-72 (45.4) > stew (41.1) > vulkan-patched (38.8)**. No-MTP king = b10181 (45.8). pp (server 443tok): ROCm 675-695 (rocm-official/rocm-72) >> vulkan ~156.

## 2. Kwaipilot KAT-Coder V2.5 Dev IQ4_XS **MTP** (19.7 GB, MoE 256/8 + MTP head) — Vulkan b10181

| Context | `-b`/`ub` | ncmoe | pp t/s | tg t/s |
|:-------:|:---------:|:-----:|:------:|:------:|
| 32K | 4096/1024 | **14** | ~2180 | **~47** |
| 64K | 4096/1024 | 14 | ~2100 | ~46 |
| 128K | 2048/1024 | 16 | ~2000 | ~44 |
| 256K | 512/512 | 18 | ~1100 | ~40 |

ROCm: **stew675 ncmoe 10/11/13/17 → tg 53-56** 🏆 (best KAT backend: pp 2600-2850, tg 55). MTP flag OFF (KAT MTP = worse: 35.4 vs 53.1 baseline; KAT + Ridge combined-spec = 45.2, still below no-MTP).

## 3. Qwen3.8-27B UD-IQ4_XS (14.3 GB, HYBRID 48 GDN + 16 attn) — Vulkan `vulkan/` (b10488)

| Context | `-b`/`ub` | pp t/s | tg t/s |
|:-------:|:---------:|:------:|:------:|
| 32K | 2048/2048 | ~589 | ~28 |
| 64K | 1024/2048 | ~589 | ~29 |
| 128K | **512/512** ⚠️ | ~122 | **~30** (b1024/2048 = 12 ❌ spill) |
| 256K | 512/512 | ~44 | ~11 ❌ (KV spill, launcher marks ❌) |

⚠️ NOT dense — **hybrid (48 GDN + 16 attention, qwen35 arch)**. Old 15.7 GB IQ4_XS didn't fit; new UD build (13.3 GiB) does. Vision: vulkan ✅ (32K 28.7 / 64K 28.9 / 128K 30.5), selfbuilt ✅ (20-27), **stew ❌ crash**.

## 4. Qwen3.8-27B Q3_K_M (13.8 GB, HYBRID) — Vulkan b10488

| Context | `-b`/`ub` | pp t/s | tg t/s |
|:-------:|:---------:|:------:|:------:|
| 32K | 2048/2048 | ~655 | ~37 |
| 64K | 1024/2048 | ~646 | ~32 |
| 128K | **512/512** ⚠️ | ~570 | **~35** (b1024/2048 = 11 ❌ spill) |
| 256K | 512/512 | — | ~11 ❌ (KV spill) |

KV = only 16 attn layers → **~18 KiB/token → 256K = 4.7 GiB (fits RAM!)** BUT model (13.8 GiB) + KV 4.7 + buffers > 15.9 VRAM → KV spills to RAM → tg ~11. **Only Ridge (12.6 GiB) works at 256K.**

## 5. Qwen3.8-27B Ridge-3.7bpw (12.6 GB, HYBRID, ALL ctx ✅) — Vulkan b10488 / stew675

| Backend | Config | pp | tg |
|---------|--------|---:|---:|
| Vulkan (no spec) | b2048/2048 @32K | 597-660 | 31.5 |
| **Vulkan + combined spec** 🏆 | b1024/128, **q4_1 KV**, MTP+ngram, fit-target 30, cache-ram 6000 | 620-660 | **39-41** (spikes 80) |
| stew675 + **adaptive MTP** 🏆 | b2048/2048, `draft-mtp-adaptive` n-min 3 / n-max 8 | 872 | **44.4** |
| stew675 / selfbuilt no-spec | b2048/2048 | 880-1030 | 28-31 |

Ridge works at **ALL contexts incl 256K** (pp 720+, tg 27-33).

- Reasoning flags: `--reasoning on --reasoning-preserve` + env `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"preserve_thinking": true, "reasoning_effort": "medium"}` (env var = batch-safe JSON; argv JSON mangled by cmd).
- Raw completion on thinking models = immediate EOS — retry needed (quirk, not crash).

## 6. Qwen3.8-9B-Distill Q8_0 (9.1 GB, HYBRID, **q8_0 KV**) — ROCm selfbuilt

| Config | pp | tg |
|--------|---:|---:|
| b512/512, q8_0 KV, no MTP | 2958 | 58.2 |
| **b512/512, q8_0 KV, `--spec-type draft-mtp`** 🏆 | 2958 | **83.4** (+43%) |

Q8_0 KV matches Q8 weights (quality pick). Q4_K_M (5.8 GB) was faster (84 / 120.4 MTP) but removed from launcher — user choice. Server test: pp 2958.5, tg 83.4 (prompt_n 885, predicted_n 200).

---

## Per-model KV rules (validated)

| Model | KV type | Result |
|-------|---------|--------|
| Qwen3.6 / KAT / Q3_K_M / IQ4_XS | q4_0 | ✅ best |
| Ridge spec (vulkan) | **q4_1** | ✅ tg 39.6 (q4_0 = 31, q5_1 = 39.0) |
| 9B Q8_0 | **q8_0** | ✅ tg 58.2 / 83.4 MTP (matches Q8 weights) |
| ANY ROCm | q5_1 | ❌ **broken** (CPU fallback) |
| KAT vulkan q4_1 | — | fine post-reset (was drift) |

## Launcher

All baked into `../run_model.bat` (model × backend × 4 contexts × MTP toggle × **vision toggle** × GPU reset × ADLX power). MoE menu: vulkan-b10181 vs selfbuilt vs stew vs **vulkan-patched** vs **rocm-72**. Dense/hybrid: vulkan vs stew vs selfbuilt. Launcher auto-kills stale PID on port 8081 before launch. Per-model KV overrides (9B → q8_0). Vision = `--mmproj <file>` toggle, offered only for models with mmproj on disk:

| Model | mmproj | 
|-------|--------|
| Qwen3.6 IQ4_XS MTP | `unsloth/.../mmproj-F32.gguf` |
| Qwen3.8 IQ4_XS / Q3_K_M | `unsloth/.../mmproj-F16.gguf` |
| Qwen3.8 Ridge | `empero-ai/.../mmproj-Qwen3.8-27B-BF16.gguf` |
| KAT / 9B Q8_0 | none on disk |

Vision API: `POST /v1/chat/completions` with `image_url` (base64 `data:` URI). Verified red-square test on all 3 (vulkan b10488). **⚠️ stew backend CRASHES on vision** (`MUL_MAT failed` — b10486 fork lacks vision encoder path) — launcher auto-skips vision on stew. NOTE: thinking models answer in `reasoning_content` (content empty) — read that field or disable thinking.
