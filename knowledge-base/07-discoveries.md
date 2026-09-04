# 07 — Key Discoveries & Pitfalls

The hard-won lessons. Read before touching anything.

## 1. MTP was "bad" — until it fit
Earlier session concluded "MTP is net negative" (tested draft-mtp n_max 1-10 → all slower). **Wrong conclusion.** The failure was VRAM: MTP draft heads (`nextn` tensors, ~400 MB) pushed ncmoe=18 over the fit point → spill → 27 t/s. At **ncmoe=20** (one more CPU layer) it fits → **61.6 t/s (+38%)**.

**Rule: MTP needs ncmoe+2, and `--spec-draft-n-max` must stay at the trained default (3).**

## 2. n_max=3 is a spike, not a knob
`--spec-draft-n-max`: 2→22.7 t/s, **3→61.6**, 4→18.9, 5→15.8. Qwen3-MTP trains exactly 3 MTP heads. Any other depth destroys acceptance. **Never change it.**

## 3. The fit-point cliff
One ncmoe below fit = **13× pp collapse** (not graceful). ROCm 32K: ncmoe=19 → 236 pp, ncmoe=20 → 3013 pp. Vulkan 32K: 17 → ~130, 18 → 2214.
Cause: VRAM overflow → weights spill to host → PCIe traffic on every prompt token.

## 4. Fit point drifts between reboots
Same binary, same config, after restart:
- ROCm 128K ncmoe=18: pre-restart 425/39.5 → post-restart **189/34.0** ❌
- ncmoe=20: pre-restart collapsed → post-restart 538/39.0 ✅
- **Vulkan: identical before/after** — no drift

**ROCm fit is a gamble per boot. Vulkan is deterministic. This alone justifies Vulkan.**

## 5. The prompt-cache pp mirage
Server caches prompt in KV. Second request with same prompt evaluates ~5 tokens → reported pp = 40 t/s. First request = real number. We chased this "collapse" for hours before reading `cache_n` in the response. **Always cold-test pp.**

## 6. ubatch > batch (the pp lever)
Raising ubatch 512→2048 = **+58-120% pp** on Vulkan. Raising batch alone = nothing. Compute buffers scale with ubatch → big ubatch eats VRAM → may force ncmoe up.

## 7. Release assets update silently — and MoE got SLOWER per build
b10181 → b10223 → … → b10453 appeared in one month. Builds also regressed: MoE server decode b10181 = 55.7 → b10243 = 49-50 → b10253 = 45 → b10338 = 45.5 → b10434/35 = 42-44 → b10453 = 36.8-40. **b10181 = MoE tg king; check the `build:` line before benchmarking.** Dense models don't have the regression.

## 8. DeltaNet pp degrades with prompt length
Qwen/Bonsai (SSM hybrid): pp1024=952 → pp10000=821. Recurrent state is sequential — long prompts can't parallelize like transformer attention. Not a config bug.

## 9. ROCm vs Vulkan — the real story
- ROCm bench pp (3013) > Vulkan bench pp (2747) — and b10435+ ROCm server pp is also strong (2800+)
- **ROCm wins pp; Vulkan wins tg + stability** (dense: Vulkan wins both — ROCm dense pp 3× worse)
- ROCm official: **i-quants crash** (`MUL_MAT invalid arg`); zips miss runtime DLLs (see 01 fix)
- ROCm fit drifts per boot; Vulkan deterministic
- Ternary (Q1_0/Q2_0) + i-quants: Vulkan only (mainline ROCm lacks kernels)

## 10. MTP heads-only GGUF merging (how we fixed the 27B draft)
A 436 MB `27B_MTP.gguf` (only `blk.64` + nextn heads) failed as a draft: first `missing qwen35.rope.dimension_sections`, then `missing tokenizer merges`, then `missing tensor token_embd.weight` — the loader demands a complete model.

**Solution — merge the MTP layer into the main model:**
1. Read main GGUF (has all metadata)
2. Copy all main KV but `block_count 64→65`, add `nextn_predict_layers=1`
3. Copy all main tensors
4. Append the draft's `blk.64.*` tensors (15 total: attn/ffn/norms + 4 nextn)
5. Result: model with built-in MTP → `--spec-type draft-mtp` works alone

**The gguf-py API lesson:** use `ReaderField.contents()` (not `.data`/`.parts`) for values, and `GGUFWriter.add_key_value(key, val, type, sub_type)` (not `add_key/add_val`). String arrays need element-type handling. `PYTHONIOENCODING=utf-8` to avoid console encode crashes.

## 11. `.bat` gotchas
- **cmd needs CRLF** line endings (write_file writes LF → "syntax of the command is incorrect")
- `if X set A & goto B` = the `goto` ALWAYS runs — use label-based branches
- `set /p` with piped input only reads the FIRST line (subsequent read empty) → use `choice` command
- `choice /c 12` + `if errorlevel 2` (errorlevel = index, ≥ semantics)
- Background servers die when the spawning shell exits — one-command test pattern

## 12. The b=4096/ub=2048 quirk
Some models collapse at this exact combo (Bonsai: 917→341 pp) while 4096/4096 and 2048/2048 are fine. Kernel-path sensitivity. **Sweep batch combos per model.**

## 13. 256K is a different world
KV 1.44 GB + compute buffers force MoE: b=512, ncmoe+5. **ROCm/HIP: any bigger batch = hard crash (device -1) at request time, not load time.** Health endpoint lies — healthy until first completion. **Vulkan: immune** (KAT IQ4_XS: b512/1024/2048 all work @256K, identical perf, pp ~1470, tg ~51).

**Dense 256K = broken everywhere** (Q38: pp 27-55 any batch, GPU pegged — attention-kernel fallback at 262K-seq). Dense KV = 36 GiB @256K → lives in system RAM (see 06). **Dense max = 128K.**

## 13b. GPU state drift & the editor tax
- After long bench sessions all backends drop ~8-14% (thermal/clock) — same-session comparisons only; `reset-gpu.ps1` (PnP cycle) restores
- **Zed (the editor) renders on the AMD GPU** — ~18% sustained, bursts to 79% on the 3D engine → pollutes benches. Close/minimize for clean numbers

## 13c. i-quants are a different world too
- IQ3_XXS etc.: bench may hang after table header → server tests only
- Vulkan: ub/quant-specific collapse zones (IQ3_XXS @128K: ONLY b1024/ub1024 works; b512 and ub2048 both collapse)
- ROCm official crashes on them; rocm-selfbuilt runs them
- 256K broken for IQ3_XXS

## 13d. chat-template-kwargs + batch JSON
- Thinking effort: `--reasoning-effort medium` (b10434+) OR env `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={...}` — **env var is the only batch-safe way to pass JSON** (argv JSON gets split into 4 args by cmd)
- Qwen3.8 template defaults to **xhigh** thinking — set medium or pay 2-3× thought tokens

## 14. mmap variance tells you it's page faults
mmap runs show ±200-270 t/s standard deviation on pp. That jitter is the fingerprint of lazy page-in. `none` is ±20-30.

## 15. ncmoe over-offload is sometimes useful
Bonsai (fits in VRAM): ncmoe=8 gives +3.8% short-prompt pp (GPU underutilized → CPU overlap helps) but −4.2% long-prompt pp and −2.2% tg. **Only when GPU is idle.**

## 16. (2026-08-20) GPU stuck-clocks poisoned ~half of all "findings"
After hours of benching, GPU clocks stick at idle → **everything drops 2-10×** (KAT 67→33, pp 2770→1200). Re-verified post-reset:
- KAT q4_1 "broken" (147/18) → actually 67.7 ✅
- Q3_K_M q4_1/q5_1 "VRAM cliff" (32/12) → actually 628/38.6 and 685/39.4 ✅
- stew/selfbuilt q5_1 "broken" → **REAL** (ROCm has no q5_1 KV kernel: `SOLVE_TRI`-style CPU fallback, pp ~65) ❌
- Many pp 1390-vs-2830 swings = same config, different state
**Rule: PnP reset before every A/B series; interleave configs; 5× retries at n≥1000; trust tg (stable) over pp (state-sensitive).**

## 17. (2026-08-20) Qwen3.8-27B is HYBRID, not dense
48 GDN (DeltaNet linear-attention) + 16 full-attention layers (3:1, verified in GGUF). KV only from 16 attn layers → **~18 KiB/token → 256K = 4.7 GiB fits RAM**. "Dense 256K broken / 36 GiB KV" = wrong (drift artifact). Ridge runs 256K fine (720/27.4).

## 18. (2026-08-20) GDN fused-op bug #27327 — fixed locally
`resolve_fused_ops` disabled fused GDN whenever layer-0 (CPU-pinned) mismatched the fused op's GPU placement → SSM fell to CPU. Local fix: disable only when op fell back to CPU while layer is GPU. **+2-6% tg MoE on ROCm.** Upstream issue open; no merged fix.

## 19. (2026-08-20) GDN clustered-columns — the one patch that worked
`lanes_per_column` (32/16/8): 2-4 state columns per RDNA4 wave. **LANES=8 default: +9% pp Qwen3.6, +3% pp Qwen3.8**, tg flat. 36/36 backend-ops correct. Baked into rocm-selfbuilt (env `GGML_HIP_GDN_LANES` still overrides).

## 20. (2026-08-20) HIP `__byte_perm` is software-emulated
HIP's `__byte_perm` = ~10 C++ ops, NOT native `v_perm_b32`. Upstream already HIP-natives Q4_0 tables; Q2_0/Q2_K swaps measured neutral (bottleneck elsewhere); Q3_K/Q4_K/Q5_K/Q6_K don't use it; Q1_0 irrelevant. pi-rdna-kernels reverted.

## 21. (2026-08-20) MTP: model-dependent, backend-dependent
- **Qwen3.6-MTP + selfbuilt + plain `draft-mtp`: tg 37.2 → 48.6 (+31%)** ✅ (best MoE MTP, 2026-08-20)
- **UPDATED (2026-08-22): vulkan MTP wins** — b10488 + plain draft-mtp n_max=3 → **51.6-56.6**, b10181 + MTP → 53.2. Old "vulkan-b10181 has no spec support" = WRONG (flag works, 53.2). Q38-style p-min 0.82/n_max=1 = poison on Qwen3.6 (40.0).
- **UPDATED (2026-08-22): stew MTP does NOT crash on qwen35moe anymore** — plain draft-mtp 41.1 / adaptive 40.6 (both 200-tok clean), but = stew baseline 41 → **neutral, no gain** on Qwen3.6 (adaptive only pays on Ridge +55%).
- KAT-MTP MTP: 35.4 vs 53.1 baseline → keep OFF
- Ridge + stew **adaptive** MTP (`draft-mtp-adaptive`, stew-only feature): 28.7 → **44.4** ✅; plain MTP on selfbuilt = 24.9 ❌
- Ridge + vulkan combined `draft-mtp,ngram-mod` + q4_1 KV + b1024/ub128 + fit-target 30 + cache-ram 6000: 31.5 → **39.6** (+30%, spikes 96) — eaman's config; big-ubatch + spec = pp collapse (63)
- **stew MTP crashes on qwen35moe** (KAT/Qwen3.6, load or mid-gen) — Ridge (dense) only
- Draft KV (`-ctkd/-ctvd q4_0`) CRASHES KAT-MTP draft graph; q4_1/q5_1 load but no gain

## 22. (2026-08-20) ROCm q5_1 KV = broken everywhere
selfbuilt + stew: q5_1 KV → `device kernel image is invalid`-class CPU fallback (pp 65-157). Vulkan q5_1 fine (39-48 tg). q4_0/q4_1 safe on ROCm.

## 23. (2026-08-20) Build env gotcha
Bare `ninja` on `llama.cpp-src/build_official` fails compiling `.cu` (Windows CRT `math.h` leaks: `ldexpf` host-call) — needs ROCm env from `tools/ninja_rocm.ps1` (vcvars + HIP_PATH/ROCM_PATH).

## 24. (2026-08-20) Stale server on port 8081 = false "crash"
A leftover `llama-server` keeps listening on 8081 after its parent shell dies. Restart then "crashes" (code 1) or health-checks pass against the OLD model (wrong KV types, timings=None from /completion). **Fix baked into launcher: `netstat -ano | findstr ":8081" | findstr LISTENING` → taskkill before launch.** Reproduced while testing 9B Q8_0 (health OK against stale server → timings None).

## 25. (2026-08-20) Q8_0 9B: quality pick, big MTP gain
Qwen3.8-9B-Distill Q8_0 (9.1 GB) + **q8_0 KV**: pp 2958, tg 58.2 → **83.4 with draft-mtp (+43%)** (selfbuilt). Q4_K_M (removed) was faster (84 / 120.4 MTP) but lower quality — Q8 weights want Q8 KV precision.

## 26. (2026-08-20) ROCm 7.2 runtime beats 7.14 for MoE tg
Same b10488 + patches: **7.2 → pp 3013 / tg 39.3 vs 7.14 → pp ~2900 / tg 35.3 (+11% MoE tg)**. Ridge slight 7.14 win (+4%). Runtime DLLs for `rocm/` copied from `C:\Program Files\AMD\ROCm\7.2\bin` (NOT `amdhip64_7.dll` — driver copy must win). `rocm-72/` = selfbuilt patches + 7.2 = best pp.

## 27. (2026-08-20) Qwen3.8-27B vision works — flag is `--mmproj`, NOT `-mmproj`
All three 27B Q38 variants load mmproj + see images (red-square test, vulkan b10488): Q3_K_M/IQ4_XS → `unsloth/.../mmproj-F16.gguf` (0.86 GB), Ridge → `empero-ai/.../mmproj-BF16.gguf` (0.87 GB). **9B has no mmproj** (27B ones die — hidden-dim mismatch). Qwen3.6 also has `mmproj-F32.gguf`. Gotcha 1: server flag is `--mmproj`/`-mm` (`-mmproj` rejected). Gotcha 2: thinking models put the answer in `reasoning_content` — `content` is empty — parse that field or use `--reasoning off`. Ridge+vision tg 34.6 (≈ no-vision 31-35, no penalty).

## 28. (2026-08-20) 27B hybrid 128K = VRAM spill at b1024/2048
IQ4_XS (13.3 GiB) + KV 2.3 GiB (128K) + compute buffers > 15.9 usable → KV/buffers spill → **tg 28 → 12**. Q3_K_M same (13.8 GiB): 11.4. **b512/512 fixes it** (IQ4_XS 30.5, Q3_K_M 34.7). Docs' old "128K = 28-31" was wrong. 256K (KV 4.7 GiB) spills regardless → tg ~11 → only Ridge (12.6 GiB) viable at 256K. **Stew backend vision = CRASH** (`MUL_MAT failed`, b10486 fork lacks Qwen3.5 vision-encoder path; text-only works pp 737/tg 30). Vulkan+selfbuilt vision OK; launcher gates stew out of the vision toggle.

## 29. (2026-09-04, Linux) `--cpu-strict` is NOT neutral like it was on Windows — now a fixed parameter
Moved to a Linux box (Arch, kernel 7.1.9) with the equivalent CPU class (i9-14900K). Discovery #(Windows, 04-parameters.md "Threads") said CPU pinning was neutral at `-t 8`. Re-tested on `rocm-moe-linux` (Qwen3.6-35B-A3B MoE, `-t 8 -ncmoe 18`, `llama-bench -p 512 -n 128 -r 3`):
- `--cpu-strict 0` (default): pp 1049.71, tg 34.89
- `--cpu-strict 1`: pp **1270.49 (+21%)**, tg 34.84 (flat) — a real, repeatable win
- `--cpu-strict 1` + pinned to 4 physical P-cores (`--cpu-mask 0xff` / `-Cr 0-7`): pp 1234.41 (no further gain), tg 35.43 with tighter variance

Follow-up: also tested dense/hybrid Qwen3.8-27B models (IQ4_XS, GSQ-RCO IQ3_S) on `rocm-linux`, `ncmoe=0`. No throughput win there (+0.4-1.8% pp, tg flat/negligible), but pp variance dropped 2.7-4x in both cases with no regression anywhere.

**Rule: on Linux, `--cpu-strict 1` is now unconditional in `run_model.sh`** — promoted from a MoE-only/ncmoe-gated flag to a fixed parameter for every model and backend, since it never hurt and helps meaningfully on MoE. This is the opposite of the Windows finding. Likely cause: Linux's CFS scheduler migrates threads across the P/E-core boundary more readily than Windows for this workload, so strict placement has more to fix. Don't hardcode a specific `--cpu-range` in shared scripts (P-core logical-ID layout is topology/BIOS-specific); `--cpu-strict 1` alone is portable and captures nearly all of the gain.
