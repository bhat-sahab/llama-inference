# llama.cpp Local Inference — AMD RX 9070 XT

Local LLM inference server using [llama.cpp](https://github.com/ggml-org/llama.cpp), optimized for AMD Radeon RX 9070 XT (RDNA4, gfx1201) with Intel i9-14900K.

## Quick Start

```powershell
python launcher.py
```

Pick a model, select a backend, click **Launch Server**, then open http://localhost:8081.

## Project Structure

```
D:\Work\llama/
├── launcher.py                  ← GUI launcher (tkinter, no dependencies needed)
├── launcher_config.json         ← Saved preferences (last model/backend)
├── benchmark.md                 ← Full benchmark results
├── README.md                    ← This file
│
├── llama.cpp/
│   ├── bin/
│   │   ├── vulkan/              ← Mainline Vulkan + CPU binaries — b10068
│   │   │   ├── llama-server.exe
│   │   │   ├── ggml-vulkan.dll
│   │   │   ├── ggml-cpu-alderlake.dll
│   │   │   ├── llama-bench.exe
│   │   │   └── ...
│   │   │
│   │   ├── rocm/                ← Mainline ROCm/HIP binaries — b10068
│   │   │   ├── llama-server.exe
│   │   │   ├── ggml-hip.dll
│   │   │   └── ...
│   │   │
│   │   └── prism-hip/           ← PrismML fork ROCm/HIP (Q2_0 kernels for Bonsai)
│   │       ├── llama-server.exe
│   │       ├── ggml-hip.dll
│   │       └── ...
│   │
│   ├── models/                  ← Model cache
│   │   └── .cache/
│   │
│   ├── templates/               ← Chat templates
│   │   ├── chat_template.jinja
│   │   └── qwen3-coder-30b-template.jinja
│   │
│   └── rocm.zip / vulkan.zip
```

## Models

The launcher ships with these pre-configured models. Models are stored in `~/.lmstudio/models/`.

| Model | Quant | Size | Disk Path |
|-------|-------|:----:|-----------|
| **Gemma-4-12B** | Q4_K_M | 6.7 GB | `unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf` |
| **Qwen3.5-9B** | Q4_K_M | 5.3 GB | `lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` |
| **Qwen3-Coder-30B-A3B** | Q4_K_M | ~18 GB | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` |
| **Ternary-Bonsai-27B** | Q2_0 | 7.3 GB | `prism-ml/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_g64.gguf` |
| **Bonsai-8B** | Q1_0 | 1.5 GB | `prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf` |

### GGUF Variants

The Ternary-Bonsai-27B model has two GGUF variants:

| Variant | Group | Size | Backend |
|---------|:-----:|:----:|---------|
| `Q2_0.gguf` | group-128 | 6.66 GB | **PrismML fork** (ROCm only) |
| `Q2_g64.gguf` | group-64 | 7.05 GB | **Mainline** Vulkan, CPU, Metal ✅ |

The launcher uses `Q2_g64` for mainline Vulkan and switches to `Q2_0` when using ROCm PrismML.

## Backends

| Backend | Binary Dir | Bonsai-27B | Bonsai-8B | Gemma/Qwen | SDK |
|---------|:----------:|:----------:|:---------:|:----------:|:---:|
| **ROCm (PrismML)** ☑️ | `llama.cpp/bin/prism-hip/` | 54 t/s ✅ | 200+ t/s | best pp | `pip install rocm-sdk` |
| **Vulkan** | `llama.cpp/bin/vulkan/` | 58 t/s ✅ | 200+ t/s | works | None |
| ROCm (mainline) | `llama.cpp/bin/rocm/` | ❌ no Q2_0 | 200+ t/s | best pp | `pip install rocm-sdk` |

- **Vulkan** is zero-setup and gives best generation speed (58 t/s for Bonsai)
- **ROCm (PrismML)** needs `rocm-sdk` but gives best prompt processing (1315 t/s for Bonsai)

### Setup

**Vulkan** — built into the GPU driver, no setup needed.

**ROCm** — requires the AMD ROCm SDK:
```powershell
pip install rocm-sdk
```
The launcher auto-detects it via `find_hip_sdk()` by looking in the Microsoft Store Python's site-packages.

## Performance

See [benchmark.md](benchmark.md) for full results. Quick summary:

| Model | ROCm pp | Vulkan pp | ROCm gen | Vulkan gen |
|-------|:-------:|:---------:|:--------:|:----------:|
| Gemma-4-12B | **2,467** | 1,752 | 54.5 | **64.6** |
| Qwen3.5-9B | **3,609** | 2,490 | 81.7 | **92.3** |
| Qwen3-Coder-30B | 274 | **529** | **30.6** | 22.6 |
| Bonsai-27B | **1,315** | 915 | 53.3 | **58.3** |

## Launcher Usage

```powershell
python launcher.py
```

The GUI provides:
- **Model selector** — choose from pre-configured models
- **Backend** — ROCm PrismML, ROCm mainline, or Vulkan
- **Mode** — Default, Long Context, Thinking (model-dependent)
- **Quick Settings** — context size, GPU layers, threads, batch size
- **Advanced** — flash attention, mlock, no-mmap, KV cache type, temperature/top_p/top_k/repetition penalty controls, chat template, tensor overrides, extra flags
- **Launch/Stop** — controls the `llama-server` process
- **Log** — live output from the server

### CLI Flags (PowerShell)

The project also includes `llama.ps1` for command-line usage:
```powershell
.\llama.ps1 Bonsai
.\llama.ps1 Bonsai -LongCtx
.\llama.ps1 QwenCoder -Vulkan
```

## Q2_0 Upstream Status

Q2_0 (ternary) quantization support is being merged into mainline llama.cpp:

| Backend | Status | PR |
|---------|:------:|:--:|
| CPU (ARM NEON + scalar) | ✅ Merged | #24448 |
| Metal | ✅ Merged | #25419 |
| Vulkan | ✅ Merged | #25430 |
| CUDA | 🔄 In review | #25707 |
| ROCm/HIP | ⏳ Follows CUDA | — |

Until CUDA/ROCm merges, use the PrismML fork binaries in `llama.cpp/bin/prism-hip/` for ROCm, or use mainline Vulkan from `llama.cpp/bin/vulkan/`.

## Links

- [llama.cpp (mainline)](https://github.com/ggml-org/llama.cpp)
- [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp)
- [Bonsai Demo](https://github.com/PrismML-Eng/Bonsai-demo)
- [Ternary-Bonsai-27B HuggingFace](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
- [ROCm SDK for Windows](https://rocm.docs.amd.com/en/latest/install/rocm.html)
