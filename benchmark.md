# Benchmark Results — llama.cpp on RX 9070 XT

**System:** AMD Radeon RX 9070 XT (16 GB VRAM) · Intel i9-14900K (32 threads)  
**Builds:** llama.cpp b10068 (mainline) · PrismML fork 38c66ad (prism-hip)  
**Date:** 2026-07-18

---

## Quick Reference — All Models (Short Prompt: pp512 / tg128)

| Model | Size | Backend | Prompt t/s | Gen t/s | Gen ms/tok |
|-------|:----:|:-------:|:----------:|:-------:|:----------:|
| **Gemma-4-12B** (Q4_K_M) | 6.7 GB | ROCm | **2,467.5** | **54.5** | 18.4 |
| | | Vulkan | 1,751.9 | **64.6** | 15.5 |
| **Qwen3.5-9B** (Q4_K_M) | 5.3 GB | ROCm | **3,609.0** | **81.7** | 12.2 |
| | | Vulkan | 2,490.1 | **92.3** | 10.8 |
| **Qwen3-Coder-30B-A3B** (Q4_K_M) | ~18 GB | ROCm | 274.4 | **30.6** | 32.7 |
| | | Vulkan | **529.2** | 22.6 | 44.3 |
| **Ternary-Bonsai-27B** (Q2_0) | 7.3 GB | ROCm PrismML | **1,314.5** | 53.3 | 18.8 |
| | | Vulkan | 915.2 | **58.3** | 17.2 |
| **Bonsai-8B** (Q1_0) | 1.5 GB | any | — | ~200+ | — |

## 10K Context (pp5783 / tg200)

| Model | Backend | Prompt t/s | Gen t/s | Gen ms/tok |
|-------|:-------:|:----------:|:-------:|:----------:|
| **Bonsai-27B** 🏆 | ROCm PrismML | **1,243.6** | 52.5 | 19.1 |
| **Bonsai-27B** | **Vulkan b10068** | 836.9 | **58.1** | 17.2 |

---

## Full Test Parameters

| Model | Batch | UBatch | KV-K | KV-V | Flash Attn | Context | Threads |
|-------|:-----:|:------:|:----:|:----:|:----------:|:-------:|:-------:|
| Gemma-4-12B | 8192 | 4096 | q8_0 | q8_0 | on | 32K | t=6 |
| Qwen3.5-9B | 8192 | 8192 | q8_0 | q8_0 | on | 128K | t=6 |
| Qwen3-Coder-30B-A3B | 4096 | 4096 | q8_0 | q8_0 | on | 32K | t=6 |
| Bonsai-27B | **2048** | **2048** | **q4_0** | **q4_0** | on | 128K | t=6 |
| Bonsai-8B | 4096 | 4096 | q8_0 | q8_0 | on | 65K | t=6 |

---

## Backend Comparison

### Prompt Processing (pp512 — higher is better)

| Model | ROCm | Vulkan | Winner |
|-------|:----:|:------:|:------:|
| Gemma-4-12B | **2,467.5** | 1,751.9 | **ROCm +41%** |
| Qwen3.5-9B | **3,609.0** | 2,490.1 | **ROCm +45%** |
| Qwen3-Coder-30B-A3B | 274.4 | **529.2** | **Vulkan +93%** |
| Bonsai-27B | **1,314.5** | 915.2 | **ROCm +44%** |

### Text Generation (tg128 — higher is better)

| Model | ROCm | Vulkan | Winner |
|-------|:----:|:------:|:------:|
| Gemma-4-12B | 54.5 | **64.6** | **Vulkan +19%** |
| Qwen3.5-9B | 81.7 | **92.3** | **Vulkan +13%** |
| Qwen3-Coder-30B-A3B | **30.6** | 22.6 | **ROCm +35%** |
| Bonsai-27B | 53.3 | **58.3** | **Vulkan +9%** |

### Key Observations

- **ROCm wins on prompt processing** for small/medium models (+40-45%)
- **Vulkan wins on text generation** for most models (+9-19%)
- **Qwen3-Coder-30B-A3B** (MoE, 3B active / 30B total) exceeds 16 GB VRAM but loads thanks to sparse activation. ROCm struggles with prompt (274 vs 529) but leads on generation (30.6 vs 22.6)
- **Bonsai-27B on Vulkan** is the best balance: 58 t/s gen, no SDK needed
- **Bonsai-27B on ROCm PrismML** is best for prompt-heavy workloads: 1315 t/s pp, needs `rocm-sdk`

---

## Bonsai Optimization Deep-Dive

The Bonsai model was optimized by tuning KV cache quantization and batch size:

| Config | KV Cache | Batch | Gen Speed | vs Baseline |
|--------|:--------:|:-----:|:---------:|:-----------:|
| **Optimal** 🏆 | **q4_0** | **2048** | **55.2 t/s** | **+175%** |
| Kv=q4_0 only | q4_0 | 4096 | 55.0 t/s | +175% |
| b=2048 only | q8_0 | 2048 | 54.9 t/s | +174% |
| kv=f16 | f16 | 4096 | 21.5 t/s | +7% |
| **Baseline** ❌ | **q8_0** | **4096** | **20.0 t/s** | — |
| b=8192 | q8_0 | 8192 | 20.0 t/s | 0% |

### Key Insight

The **KV cache quantization** is the dominant bottleneck for Q2_0 ternary models on this GPU. Switching from q8_0 to **q4_0** KV cache doubles memory bandwidth efficiency, yielding a **2.75× speedup** (~55 vs ~20 t/s). Since the model weights are already 2-bit ternary, q4_0 KV cache is more than sufficient precision.

---

## Backend Requirements

| Backend | Bonsai-27B | Bonsai-8B | Gemma/Qwen | SDK Required |
|---------|:----------:|:---------:|:----------:|:------------:|
| **ROCm (PrismML)** | 54 t/s ✅ | 200+ t/s | best pp | `pip install rocm-sdk` |
| **Vulkan (b10068)** | 58 t/s ✅ | 200+ t/s | works | None |
| ROCm (mainline) | ❌ no Q2_0 | 200+ t/s | best pp | `pip install rocm-sdk` |

---

## Build Info

- **Mainline (Vulkan/ROCm):** `571d0d540` (b10068) — from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **PrismML fork (ROCm):** `38c66ad` (prism-b9594) — from [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
- **Q2_0 upstream status:** Vulkan ✅ merged | CUDA 🔄 in review (#25707) | ROCm ⏳ follows CUDA
