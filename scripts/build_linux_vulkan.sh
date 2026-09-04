#!/usr/bin/env sh
# Linux Vulkan build for llama.cpp (RX 9070 XT / gfx1201 via RADV).
# Linux counterpart of build_vulkan_maple.ps1 — hermetic: uses the vendored
# toolchain (tools/linux) and vendored Vulkan/SPIRV headers (vendor/), so no
# system packages are required. Output lands in backends/bin/vulkan-linux/.
set -e

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="$ROOT/llama.cpp-src"
BUILD="$SRC/build-linux-vulkan"
TARGET="$ROOT/backends/bin/vulkan-linux"
CMAKE="$ROOT/tools/linux/cmake-4.4.3-linux-x86_64/bin/cmake"
NINJA="$ROOT/tools/linux/ninja"
JOBS="$(nproc 2>/dev/null || echo 16)"

# External-project steps (vulkan-shaders-gen) resolve ninja via PATH.
export PATH="$ROOT/tools/linux:$PATH"

[ -x "$CMAKE" ] || { echo "cmake not found: $CMAKE (download it into tools/linux/ or install cmake+ninja via pacman)"; exit 1; }

"$CMAKE" -B "$BUILD" -S "$SRC" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DGGML_VULKAN=ON \
    -DVulkan_INCLUDE_DIR="$ROOT/vendor/vulkan-include" \
    "-DSPIRV-Headers_DIR=$ROOT/vendor/SPIRV-Headers-install/share/cmake/SPIRV-Headers"

"$CMAKE" --build "$BUILD" --target llama-server llama-cli llama-bench llama-quantize llama-gguf-split -j "$JOBS"

mkdir -p "$TARGET"
cp "$BUILD"/bin/llama-server "$BUILD"/bin/llama-cli "$BUILD"/bin/llama-bench \
   "$BUILD"/bin/llama-quantize "$BUILD"/bin/llama-gguf-split "$TARGET"/
cp "$BUILD"/bin/*.so* "$TARGET"/

echo "Done -> $TARGET"
echo "Test: $TARGET/llama-bench --list-devices"
