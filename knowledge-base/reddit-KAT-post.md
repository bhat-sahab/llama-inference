# Kwaipilot KAT-Coder V2.5 Dev on an RX 9070 XT (16 GB) — a full tuning saga

**Tl;dr:** KAT-Coder is absurdly fast for a 27B-class MoE. My best config pushed it to **141 t/s generation / 3325 t/s prompt** (IQ2_M, Vulkan), and even at a quality-friendly quant (IQ4_XS, 17.5 GB) it sustains **~66 t/s tg / 3169 t/s pp** on 16 GB VRAM. The entire magic is one flag: `--n-cpu-moe`, which lets you keep ~85% of the weights (the experts) on CPU RAM while the GPU only ever touches the 8 active experts.

---

## Setup

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, 16 GB VRAM (gfx1201, RDNA4) |
| CPU | Intel i9-14900K (8P+16E, 32 threads) |
| RAM | 32 GB |
| OS | Windows 11 |
| Backends | llama.cpp **Vulkan** (b10243) · **ROCm self-built** (b10253, Clang 23) |

## The model

- **Kwaipilot KAT-Coder V2.5 Dev** — `qwen35moe` architecture
- 40 layers, **256 experts, 8 active per token** (plus always-on shared expert)
- Gated DeltaNet + attention hybrid (SSM), 262K native context
- Tested quants: IQ2_M (11.2 GB, 2.7 bpw), IQ4_XS (17.5 GB, 4.3 bpw), IQ4_NL (19 GB)

The key trick for fitting these on 16 GB: expert weights are ~85% of the file but only 8/256 ever run per token, so offloading them to CPU RAM costs almost nothing.

## Results — best config per quant (Vulkan, clean state)

| Quant | ncmoe | pp2048 t/s | tg200 t/s |
|:------|:-----:|:----------:|:---------:|
| **IQ2_M** | 0 | **3325** | **141** 🏆 |
| **IQ4_XS** | 9 | 3169 | **66.6** |
| **IQ4_NL** | 13 | 3070 | 56.6 |

`ncmoe` = number of MoE layers kept on CPU. It is the **VRAM fit dial** — this is how a 17-19 GB model runs on a 16 GB card.

## Per-context tuning (Vulkan, IQ4_XS — the config I settled on)

| Context | b/ub | ncmoe | tg t/s |
|:-------:|:----:|:-----:|:------:|
| 32K | 4096/1024 | 9 | ~66 |
| 64K | 4096/1024 | 10 | ~61 |
| 128K | 2048/1024 | 11 | ~58 |
| 256K | 512/512 | 14 | ~52 |

Rule of thumb: **ubatch drives prompt speed, ncmoe drives generation speed, and every +1 context step needs +1 ncmoe** (KV cache eats VRAM, so more experts move to CPU).

## The discoveries (the fun part)

### 1. The IQ4 slow-kernel trap 🪤
At `ncmoe=9` IQ4_XS generates at **46.5 t/s**. Bump to `ncmoe=10` → **58.5 t/s (+26%)**. Same model, same GPU — the difference is which kernels run where. Below a magic offload threshold, llama.cpp routes IQ4 expert matmuls to a slow Vulkan/ROCm kernel; past it, CPU AVX2 takes over and is *faster* than the GPU kernel. Same trap exists on ROCm (threshold ncmoe=10 there).

### 2. `--n-cpu-moe` beats `-ngl` for big MoE
`-ngl 99` puts everything it can on GPU, but for MoE the smart move is: keep **attention/embedding on GPU, experts on CPU**. One flag below the fit point and prompt processing collapses ~13×; one above and you're leaving free tg on the table.

### 3. `--load-mode none` is mandatory
`mmap` costs **2.7× prompt processing** on big models. Loading straight into RAM (no mmap, no mlock) is the fastest cold start and fastest pp.

### 4. q4_0 KV cache everywhere
On DeltaNet/SSM hybrids, q4_0 KV is both the fastest and safe. On my ternary Bonsai-27B it was worth **2.75× tg** vs q8_0.

### 5. Vulkan > ROCm on this card
ROCm (self-built b10253) gets close — IQ4_XS @32K: **58.5 t/s** vs Vulkan 66.6 — but Vulkan is faster on generation, doesn't drift between boots, and needs no SDK. ROCm also collapses harder at long context (128K: 44.5 vs 58).

## Real-world server session (128K ctx, 27 requests, 25k tokens)

Not a benchmark — an actual usage session from the server log:

| Metric | Value |
|--------|-------|
| tg avg | **55.6 t/s** (51–61) |
| tg early → late session | 59.7 → 54.4 (−9%, KV growth tax) |
| pp, 14.5K-token prompt | **2099 t/s** |
| pp, 65.9K-token prompt | **1460 t/s** (45 s) |
| Errors / crashes | none |

Sustained ~55 t/s at 128K context on a 16 GB card, running a 17.5 GB model, is genuinely good for local hardware.

### IQ4_NL (19 GB) — the quality pick

Ran the exact same 27-request workload on IQ4_NL (4.3 → 4.75 bpw, +1.5 GB, needs ncmoe 13-15 vs 9-11):

| Metric | IQ4_XS | IQ4_NL |
|--------|:---:|:---:|
| tg avg @128K | **55.6** | 51.1 (−8.8%) |
| tg range | 51–61 | 49.4–52.5 (tighter) |
| tg early → late | 59.7 → 54.4 | 52.4 → 49.8 |
| pp, 4.5K prompt | 1369–2158 | 1369–2158 (identical) |
| errors | none | none |

- The −8.8% tg is exactly the bench delta (56.6 vs 66.6) — the price of more weights on CPU
- NL is noticeably steadier (tight range, less KV-growth decay)
- Per-context: ncmoe 13/13/15/18 (32K/64K/128K/256K) vs XS's 9/10/11/14
- Verdict: XS if speed matters, NL if you want the extra quant quality — pp is a wash either way

## Bonus: HauhauCS Qwen3.6-27B (dense) + merged MTP — the contrast test

Same GPU, same workflow — but this 27B is **fully dense** (0 experts, verified from GGUF tensors: only `ffn_up/gate/down`, no `*_exps`). Two big lessons from trying to make it fast:

| | KAT IQ4_XS (MoE) | HauhauCS IQ3_XS (dense + MTP) |
|---|---:|---:|
| Size | 17.5 GB | 12 GB |
| pp @128K | 1300-2100 | 535-688 |
| tg @128K | **55.6** | **49.6** |
| ncmoe | 11 (crucial) | 0 (no-op — no experts) |

- **`--n-cpu-moe` is useless on dense models** — there are no experts to offload. A pp "gain" I attributed to it turned out to be noise.
- **MTP head merge** (speculative decoding with the model's own next-token head) gave **+86% tg** (33.4 → 62.1) at 32K.
- **⚠️ Silent VRAM collapse at long context**: @128K with `b=4096/ub=2048` + MTP, the server *silently* degrades — pp 553→53, tg 57→15.5 — no error, no crash, healthy health endpoint. Drop to `b=512/512` and it's fine (pp ~600, tg ~50). Same flags @32K are perfect. **Long-context MTP configs must be re-tested per context size.**
- **⚠️ temp 0.0 + MTP quirk**: greedy sampling with draft-MTP sometimes returns exactly 1 token (stop_reason null) on long prompts. temp 0.8+ generates normally. Benchmark scripts need temp ≥ 0.5 with MTP models.
- Dense 27B is inherently slower than 8/256-expert MoE on pp (767 vs 3169) — every token pays the full 27B.

## Pitfalls I hit (so you don't)

- **GPU stuck at idle clocks**: mid-session tg collapsed 32 → 2.9 t/s. Not thermal, not drift — the driver wedges at idle clocks after heavy MoE churn. Fix: PnP disable/enable the GPU (no reboot), or option 6 in my launcher.
- **Prompt-cache artifact**: a second request with the same prompt looks 10× slower on pp — it only evaluates the tail. Always cold-test pp on a fresh server.
- **llama-bench ≠ server**: bench uses minimal compute buffers; the server reserves per-slot + context KV, so configs that work in bench can collapse in the server (KAT IQ4_XS needs ub=1024 in server, 2048 was fine in bench).
- **256K context**: **ROCm/HIP hard-crashes (device −1) at request time with any batch > 512** — the health endpoint lies, it dies on the first completion. Vulkan is immune: b512→b2048 all work with identical perf (~1470 pp / ~51 tg on KAT XS) — KV size dominates at 256K, not batch. Launcher uses b=512/ub=512 as the safe universal setting.
- **Long-context + MTP + big batch = silent collapse**: the HauhauCS MTP model lost 90% pp and 75% tg at 128K with b4096/2048 — no error, healthy health endpoint. Same flags at 32K are fine. Re-validate per context; the KV cache growth is what tips it over.

## Full config that runs everything

```
llama-server -m KAT-IQ4_XS.gguf -c 131072 -t 8 -b 2048 --ubatch-size 1024
  -ctk q4_0 -ctv q4_0 --flash-attn on --load-mode none
  --n-cpu-moe 11 --cache-ram 0 --no-repack --fit off -np 1
```

## Tooling (Windows)

- `run_model.bat` — interactive launcher: 5 models × Vulkan/ROCm × 4 context sizes, per-model pp/tg display, MTP toggle, built-in GPU reset (PnP cycle)
- `bench.py` — llama-bench wrapper with crash detection (exit-code decoding, GPU TDR event monitoring, sweep abort-on-crash)
- `server_test.py` — spawn server → health-poll → real completion → pp/tg/spec-stats
- `log_parse.py` — parse any `-lv 3` server log into pp/tg stats

## Verdict

For coding on 16 GB VRAM, KAT-Coder V2.5 Dev (IQ4_XS or IQ2_M if you want raw speed) + Vulkan + `--n-cpu-moe` is the best local setup I've found this year — 2-3× faster generation than any dense model of comparable size, and the MoE offloading trick means VRAM is never the wall. Happy to answer questions about the configs, the PnP reset trick, or the GGUF expert-count experiments.

---
*Hardware: RX 9070 XT 16 GB · i9-14900K · 32 GB RAM · Windows 11 · llama.cpp Vulkan b10243 / ROCm self-build b10253*
