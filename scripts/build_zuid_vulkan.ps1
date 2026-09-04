# Build LaurentZuijdwijk llama.cpp fork (b10641, adaptive spec + Vulkan tuning) -> backends/bin/vulkan-zuid
$ErrorActionPreference = "Stop"
$NINJA = "C:\Users\BhatSahab\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
$VS = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"
$SRC = "D:/Work/llama/llama.cpp-zuid"
$BUILD = "$SRC/build-vulkan"
$TargetDir = "D:/Work/llama/backends/bin/vulkan-zuid"

if (-not (Test-Path $SRC)) { throw "Source not found: $SRC" }

cmd /c "`"$VS/VC/Auxiliary/Build/vcvarsall.bat`" amd64 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process") }
}

# Use the newest installed Vulkan SDK (installed via winget: KhronosGroup.VulkanSDK).
$VulkanSdk = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if (-not $VulkanSdk) { throw "Vulkan SDK not found under C:\VulkanSDK - install with: winget install KhronosGroup.VulkanSDK" }
$env:VULKAN_SDK = $VulkanSdk.FullName
$env:Path = "$env:VULKAN_SDK\Bin;$env:Path"
Write-Host "Vulkan SDK: $env:VULKAN_SDK" -ForegroundColor Cyan

Write-Host "Configuring zuid fork (Vulkan)..." -ForegroundColor Cyan
cmake -B $BUILD -S $SRC -G "Ninja" `
    -DCMAKE_MAKE_PROGRAM="$NINJA" `
    -DGGML_VULKAN=ON `
    -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

Write-Host "Building..." -ForegroundColor Cyan
cmake --build $BUILD -j
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
Get-ChildItem "$BUILD/bin" -Filter "*.exe" | ForEach-Object { Copy-Item $_.FullName $TargetDir -Force }
Get-ChildItem "$BUILD/bin" -Filter "*.dll" | ForEach-Object { Copy-Item $_.FullName $TargetDir -Force }
Write-Host "Done: $TargetDir" -ForegroundColor Green
