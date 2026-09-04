# Incremental rebuild of llama.cpp-src build_official with ROCm env, no clean/copy.
$ROOT = Split-Path $PSScriptRoot -Parent
$BASE = Join-Path $ROOT ".venv\Lib\site-packages"
$ROCM_CORE = "$BASE\_rocm_sdk_core"
$ROCM_LIBS = "$BASE\_rocm_sdk_libraries"
$ROCM_DEVEL = "$BASE\_rocm_sdk_devel"
$NINJA = "C:\Users\BhatSahab\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
$VS = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"

cmd /c "`"$VS/VC/Auxiliary/Build/vcvarsall.bat`" amd64 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process") }
}

$env:Path = "$ROCM_CORE\bin;$ROCM_CORE\lib\llvm\bin;$ROCM_LIBS\bin;$ROCM_DEVEL\bin;$ROCM_DEVEL\lib\llvm\bin;$env:Path"
$env:HIP_PATH = $ROCM_DEVEL
$env:ROCM_PATH = $ROCM_DEVEL
$env:CMAKE_PREFIX_PATH = "$ROCM_DEVEL\lib\cmake\hip;$ROCM_DEVEL\lib\cmake\hipblas;$ROCM_DEVEL\lib\cmake\rocblas;$ROCM_CORE;$ROCM_LIBS;$ROCM_DEVEL"

& $NINJA -C "$ROOT/llama.cpp-src/build_official"
exit $LASTEXITCODE
