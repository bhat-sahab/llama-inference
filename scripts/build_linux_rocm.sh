#!/usr/bin/env sh
# Linux ROCm (HIP) build for llama.cpp (RX 9070 XT / gfx1201).
# Linux counterpart of build_official.ps1. Requires ROCm installed via:
#   sudo pacman -S --needed rocm-core rocm-llvm rocm-device-libs comgr rocm-cmake \
#     hsa-rocr hip-runtime-amd hipblas hipblas-common rocblas rocwmma rocsolver \
#     roctracer rocminfo rocprofiler-register
# Output lands in backends/bin/rocm-linux/.
set -e

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="$ROOT/llama.cpp-src"
BUILD="$SRC/build-linux-rocm"
TARGET="$ROOT/backends/bin/rocm-linux"
CMAKE="$ROOT/tools/linux/cmake-4.4.3-linux-x86_64/bin/cmake"
NINJA="$ROOT/tools/linux/ninja"
JOBS="$(nproc 2>/dev/null || echo 16)"

ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export ROCM_PATH HIP_PATH CMAKE_PREFIX_PATH=/opt/rocm
export PATH="/opt/rocm/bin:$ROOT/tools/linux:$PATH"

[ -x /opt/rocm/bin/hipcc ] || { echo "hipcc not found - install ROCm first (see header)"; exit 1; }
[ -x "$CMAKE" ] || { echo "cmake not found: $CMAKE"; exit 1; }

"$CMAKE" -B "$BUILD" -S "$SRC" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1201 \
    -DGGML_HIPBLAS=ON \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_MMQ_MFMA=ON \
    -DGGML_HIP_NO_VMM=ON

"$CMAKE" --build "$BUILD" --target llama-server llama-cli llama-bench llama-quantize llama-gguf-split -j "$JOBS"

mkdir -p "$TARGET"
cp "$BUILD"/bin/llama-server "$BUILD"/bin/llama-cli "$BUILD"/bin/llama-bench \
   "$BUILD"/bin/llama-quantize "$BUILD"/bin/llama-gguf-split "$TARGET"/
cp "$BUILD"/bin/*.so* "$TARGET"/

echo "Done -> $TARGET"
echo "Test: $TARGET/llama-bench --list-devices"
