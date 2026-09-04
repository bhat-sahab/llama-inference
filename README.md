# llama-inference — RX 9070 XT local inference lab

Tuned `llama-server` setups for **AMD Radeon RX 9070 XT** (16 GB, RDNA4/gfx1201) on Linux and Windows,
covering Qwen3.6-35B-A3B (MoE), KAT-Coder, and the Qwen3.8-27B/9B hybrid family.

This repo holds the **launcher, benchmark harness, patches, and findings**. The multi-GB build
outputs and llama.cpp source trees are deliberately untracked — see [Layout](#layout).

## Quick start

From the repository root:

```bash
python scripts/run_model.py              # interactive menu: model → backend → vision → context → MTP
python scripts/run_model.py list         # show all models/backends with on-disk status
python scripts/run_model.py run  --model ridge --backend vulkan --ctx 32768
python scripts/run_model.py test --model q36  --backend moe    --ctx 32768 --tokens 64
```

Shell and Windows launchers are also available at `scripts/run_model.sh` and
`scripts/run_model.bat`; use `python scripts/run_model.py gui` for the graphical launcher.

`test` boots the server, polls until ready, runs a fixed prompt, and reports **measured** pp/tg
next to the expected numbers baked into the model table.

## Start here

**[`knowledge-base/`](knowledge-base/README.md)** is the real output of this project — validated
configs, VRAM math, and the pitfalls that invalidated earlier conclusions.

| Doc | Covers |
|---|---|
| [02-best-configs.md](knowledge-base/02-best-configs.md) | **Final configs** per model × context × backend |
| [07-discoveries.md](knowledge-base/07-discoveries.md) | Key insights & traps — read this first |
| [06-vram-math.md](knowledge-base/06-vram-math.md) | Memory budget calculations |
| [08-testing-new-models.md](knowledge-base/08-testing-new-models.md) | Onboarding a new model |

Two rules that cost the most time to learn:

- **GPU clock drift poisons A/B runs.** After long sessions clocks stick at idle and pp/tg collapse
  2–10×. Several "broken config" conclusions were artifacts of this. Reset before any clean A/B.
- **`--n-cpu-moe N` is a cliff, not a slope.** One layer below fit = ~13× prefill collapse.

## Layout

| Path | Tracked | Contents |
|---|:---:|---|
| `scripts/run_model.py` / `.sh` / `.bat` | ✅ | Launcher + benchmark harness (`run_model_ui.py` = GUI) |
| `knowledge-base/` | ✅ | Benchmark findings and configs |
| `scripts/build_*` | ✅ | Per-backend Linux and Windows build scripts |
| `scripts/_dflash_*.sh`, `scripts/_moe_ab.sh` | ✅ | Current A/B experiments (DFlash2 draft + ngram speculative decoding) |
| `patches/` | ✅ | RDNA4 patches: GDN clustered-columns, mmvq dynamic-warps, adaptive spec |
| `tools/` | ✅ | `server_test.py`, `vision_test.py`, `mtp_test.py`, GPU reset/monitor, ADLX power |
| `traces/` | ✅ | Raw benchmark logs behind the knowledge base |
| `backends/bin/` | ❌ | Built `llama-server` per backend — rebuild via `scripts/` |
| `llama.cpp-{src,perf,stew675,rdna-boosts}/` | ❌ | Upstream + fork clones, each its own git repo |
| `vendor/`, `tools/linux/` | ❌ | Vulkan/SPIRV headers + bundled CMake/Ninja toolchain |
| `models` | ❌ | Symlink to the external LM Studio model store |

### Rebuilding a backend

```bash
scripts/build_linux_vulkan.sh     # → backends/bin/vulkan-linux
scripts/build_linux_rocm.sh       # → backends/bin/rocm-linux
scripts/build_linux_moe.sh        # → backends/bin/rocm-moe-linux   (perf fork + MoE cache)
scripts/build_linux_stew675.sh    # → backends/bin/rocm-stew675-linux
```

Build scripts use the pinned toolchain in `tools/linux/` (CMake + Ninja) and expect ROCm at
`/opt/rocm`. Clone the matching source tree next to this repo before building.

## Hardware baseline

RX 9070 XT 16 GB (gfx1201) · i9-14900K (8 P-cores, `-t 8`) · 32 GB RAM.
