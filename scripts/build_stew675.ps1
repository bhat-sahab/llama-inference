# Build llama.cpp rdna-boosts patches (baseline/d222767c7) with ROCm 7.14 pip SDK (.venv) -> backends/bin/rocm-stew675
$ErrorActionPreference = "Stop"
$BASE = "D:\Work\llama\.venv\Lib\site-packages"
$ROCM_CORE = "$BASE\_rocm_sdk_core"
$ROCM_LIBS = "$BASE\_rocm_sdk_libraries"
$ROCM_DEVEL = "$BASE\_rocm_sdk_devel"
$NINJA = "C:\Users\BhatSahab\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
$RC = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe"
$VS = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"
$SRC = "D:/Work/llama/llama.cpp-stew675"
$BUILD = "$SRC/build_stew"
$TargetDir = "D:/Work/llama/backends/bin/rocm-stew675"

cmd /c "`"$VS/VC/Auxiliary/Build/vcvarsall.bat`" amd64 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process") }
}

$env:Path = "$ROCM_CORE\bin;$ROCM_LIBS\bin;$ROCM_DEVEL\bin;$ROCM_DEVEL\lib\llvm\bin;$env:Path"
$env:HIP_PATH = $ROCM_DEVEL
$env:ROCM_PATH = $ROCM_DEVEL
$env:CMAKE_PREFIX_PATH = "$ROCM_DEVEL\lib\cmake\hip;$ROCM_DEVEL\lib\cmake\hipblas;$ROCM_DEVEL\lib\cmake\rocblas;$ROCM_CORE;$ROCM_LIBS;$ROCM_DEVEL"

Write-Host "Configuring stew675 rdna-boosts (ROCm 7.14 pip SDK)..." -ForegroundColor Cyan
cmake -B $BUILD -S $SRC -G "Ninja" `
    -DCMAKE_MAKE_PROGRAM="$NINJA" `
    -DCMAKE_C_COMPILER="$ROCM_DEVEL\lib\llvm\bin\clang.exe" `
    -DCMAKE_CXX_COMPILER="$ROCM_DEVEL\lib\llvm\bin\clang++.exe" `
    -DCMAKE_RC_COMPILER="$RC" `
    -DGGML_HIP=ON -DGPU_TARGETS="gfx1201" `
    -DGGML_HIPBLAS=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON `
    -DGGML_NATIVE=1 -DGGML_RPC=1 `
    -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600" `
    -DCMAKE_CXX_FLAGS="--rocm-path=$ROCM_DEVEL/lib/llvm" `
    -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) { Write-Host "CMake failed!" -ForegroundColor Red; exit 1 }

cmake --build $BUILD -j

if ($LASTEXITCODE -ne 0) { Write-Host "Build failed!" -ForegroundColor Red; exit 1 }

New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
Get-ChildItem "$BUILD/bin" -Filter "*.exe" | ForEach-Object { Copy-Item $_.FullName $TargetDir -Force }
Get-ChildItem "$BUILD/bin" -Filter "*.dll" | ForEach-Object { Copy-Item $_.FullName $TargetDir -Force }

# runtime DLLs from the ROCm 7.14 pip SDK (NOT amdhip64_7.dll - driver's System32 copy must win)
Copy-Item "$ROCM_LIBS\bin\hipblas.dll","$ROCM_LIBS\bin\libhipblaslt.dll","$ROCM_LIBS\bin\rocblas.dll","$ROCM_LIBS\bin\rocsolver.dll" $TargetDir -Force
Copy-Item "$ROCM_LIBS\bin\hipblaslt" $TargetDir -Recurse -Force
Copy-Item "$ROCM_LIBS\bin\rocblas" $TargetDir -Recurse -Force
Write-Host "Done: $TargetDir" -ForegroundColor Green
