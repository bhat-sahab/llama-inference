$NINJA = "C:\Users\BhatSahab\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
$VS = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"
$SRC = "D:/Work/llama/llama.cpp-maple"
$BUILD = "$SRC/build-vulkan"

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

Write-Host "Configuring deepgrove fork (Vulkan)..." -ForegroundColor Cyan
cmake -B $BUILD -S $SRC -G "Ninja" `
    -DCMAKE_MAKE_PROGRAM="$NINJA" `
    -DGGML_VULKAN=ON `
    -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) { Write-Host "CMake failed!" -ForegroundColor Red; exit 1 }

Write-Host "Building llama-server llama-cli llama-bench..." -ForegroundColor Cyan
cmake --build $BUILD -j --target llama-server llama-cli llama-bench

if ($LASTEXITCODE -ne 0) { Write-Host "Build failed!" -ForegroundColor Red; exit 1 }

$TargetDir = "D:/Work/llama/backends/bin/vulkan-maple"
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
Get-ChildItem "$BUILD/bin" -Include "*.exe","*.dll" | ForEach-Object {
    Copy-Item $_.FullName $TargetDir -Force
}
Write-Host "Done! $(Get-ChildItem $TargetDir | Measure-Object | Select-Object -ExpandProperty Count) files" -ForegroundColor Green
