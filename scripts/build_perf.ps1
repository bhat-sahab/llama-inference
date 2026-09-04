param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$Root = Split-Path $PSScriptRoot -Parent
$SourceDir = Join-Path $Root "llama.cpp-moe"
$BuildDir = Join-Path $SourceDir "build_moe"
$TargetDir = Join-Path $Root "backends\bin\rocm-moe"

# Locate ROCm packages in the project venv (.venv), falling back to the active interpreter.
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
$Base = (& $Python -c "import site; from pathlib import Path; print(next((p for p in site.getsitepackages() + [site.getusersitepackages()] if (Path(p) / '_rocm_sdk_core').exists()), ''))").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Base) -or -not (Test-Path $Base)) {
    throw "Could not locate ROCm Python site-packages for $Python"
}

$RocmCore = Join-Path $Base "_rocm_sdk_core"
$RocmLibs = Join-Path $Base "_rocm_sdk_libraries"
$RocmDevel = Join-Path $Base "_rocm_sdk_devel"

Write-Host "Python: $Python" -ForegroundColor Cyan
Write-Host "ROCm packages: $Base" -ForegroundColor Cyan

# Expand ROCm devel archive and link installed device packages.
Write-Host "Initializing ROCm SDK..." -ForegroundColor Cyan
& $Python -m rocm_sdk init
if ($LASTEXITCODE -ne 0) {
    throw "rocm-sdk init failed"
}

if (-not (Test-Path $RocmDevel)) {
    throw "ROCm development tree missing: $RocmDevel"
}

$Ninja = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
$Rc = "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe"
$Vs = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"

if (-not (Test-Path $Ninja)) { throw "Ninja not found: $Ninja" }
if (-not (Test-Path $Rc)) { throw "Windows resource compiler not found: $Rc" }
if (-not (Test-Path "$Vs\VC\Auxiliary\Build\vcvarsall.bat")) {
    throw "Visual Studio vcvarsall.bat not found under: $Vs"
}

if ($Clean) {
    Write-Host "Cleaning build and output directories..." -ForegroundColor Yellow
    foreach ($path in @($BuildDir, $TargetDir)) {
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force
        }
    }
}

# Import Visual Studio compiler environment into this PowerShell process.
cmd /c "`"$Vs/VC/Auxiliary/Build/vcvarsall.bat`" amd64 && set" |
    ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }

$PathRoots = @(
    "$RocmCore\bin",
    "$RocmCore\lib\llvm\bin",
    "$RocmLibs\bin",
    "$RocmDevel\bin",
    "$RocmDevel\lib\llvm\bin",
    "$env:SystemRoot\System32\downlevel"
) | Where-Object { Test-Path $_ }
$env:Path = ($PathRoots + $env:Path) -join ";"
$env:HIP_PATH = $RocmDevel
$env:ROCM_PATH = $RocmDevel
$env:CMAKE_PREFIX_PATH = @(
    "$RocmDevel\lib\cmake\hip",
    "$RocmDevel\lib\cmake\hipblas",
    "$RocmDevel\lib\cmake\rocblas",
    $RocmCore, $RocmLibs, $RocmDevel
) -join ";"

Write-Host "Configuring gfx1201 build..." -ForegroundColor Cyan
cmake -B $BuildDir -S $SourceDir -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$Ninja" `
    "-DCMAKE_C_COMPILER=$RocmCore\lib\llvm\bin\clang.exe" `
    "-DCMAKE_CXX_COMPILER=$RocmCore\lib\llvm\bin\clang++.exe" `
    "-DCMAKE_RC_COMPILER=$Rc" `
    "-DGGML_HIP=ON" `
    "-DGPU_TARGETS=gfx1201" `
    "-DGGML_HIPBLAS=ON" `
    "-DGGML_HIP_ROCWMMA_FATTN=ON" `
    "-DGGML_HIP_GRAPHS=ON" `
    "-DGGML_HIP_MMQ_MFMA=ON" `
    "-DGGML_HIP_NO_VMM=ON" `
    "-DCMAKE_CXX_FLAGS=--rocm-path=$RocmDevel\lib\llvm" `
    "-DCMAKE_BUILD_TYPE=Release"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

Write-Host "Building..." -ForegroundColor Cyan
cmake --build $BuildDir --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
$BuildBin = Join-Path $BuildDir "bin"
if (-not (Test-Path $BuildBin)) { throw "Build output missing: $BuildBin" }

Write-Host "Copying built binaries..." -ForegroundColor Cyan
Get-ChildItem $BuildBin -Filter "*.exe" -File | ForEach-Object {
    Copy-Item $_.FullName $TargetDir -Force
    Write-Host "  [EXE] $($_.Name)" -ForegroundColor Green
}
Get-ChildItem $BuildBin -Filter "*.dll" -File | ForEach-Object {
    Copy-Item $_.FullName $TargetDir -Force
}

# Copy runtime DLLs beside the executable. First source wins.
$RuntimeRoots = @(
    "$RocmCore\bin",
    "$RocmLibs\bin",
    "$RocmDevel\bin",
    "$env:SystemRoot\System32\downlevel"
) | Where-Object { Test-Path $_ }
$RuntimePatterns = @(
    "amdhip64_*.dll", "hipblas*.dll", "libhipblas*.dll", "rocblas*.dll",
    "hiprtc*.dll", "amd_comgr*.dll", "hsa-runtime*.dll", "libomp*.dll",
    "api-ms-win-crt-*.dll", "ucrtbase.dll"
)
$CopiedDlls = @{}
Get-ChildItem $TargetDir -Filter "*.dll" -File -ErrorAction SilentlyContinue |
    ForEach-Object { $CopiedDlls[$_.Name] = $true }

foreach ($root in $RuntimeRoots) {
    foreach ($pattern in $RuntimePatterns) {
        Get-ChildItem $root -Filter $pattern -File -ErrorAction SilentlyContinue |
            ForEach-Object {
                if (-not $CopiedDlls.ContainsKey($_.Name)) {
                    Copy-Item $_.FullName $TargetDir -Force
                    $CopiedDlls[$_.Name] = $true
                    Write-Host "  [SDK] $($_.FullName) -> $TargetDir\$($_.Name)" -ForegroundColor Yellow
                }
            }
    }
}

# Copy remaining ROCm DLLs too; package contents vary by ROCm release.
foreach ($root in @("$RocmCore\bin", "$RocmLibs\bin", "$RocmDevel\bin")) {
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem $root -Filter "*.dll" -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            if (-not $CopiedDlls.ContainsKey($_.Name)) {
                Copy-Item $_.FullName $TargetDir -Force
                $CopiedDlls[$_.Name] = $true
                Write-Host "  [SDK-ALL] $($_.FullName) -> $TargetDir\$($_.Name)" -ForegroundColor DarkYellow
            }
        }
}

# Copy GPU kernel/data libraries required by rocBLAS and hipBLASLt.
$RocblasDest = Join-Path $TargetDir "rocblas\library"
$HipblasltDest = Join-Path $TargetDir "hipblaslt"
foreach ($source in @("$RocmDevel\bin\rocblas\library", "$RocmLibs\bin\rocblas\library")) {
    if (Test-Path $source) {
        New-Item -ItemType Directory -Path $RocblasDest -Force | Out-Null
        Copy-Item (Join-Path $source "*") $RocblasDest -Recurse -Force
        Write-Host "  [LIB] $source -> $RocblasDest" -ForegroundColor Yellow
    }
}
foreach ($source in @("$RocmDevel\bin\hipblaslt", "$RocmLibs\bin\hipblaslt")) {
    if (Test-Path $source) {
        New-Item -ItemType Directory -Path $HipblasltDest -Force | Out-Null
        Copy-Item (Join-Path $source "*") $HipblasltDest -Recurse -Force
        Write-Host "  [LIB] $source -> $HipblasltDest" -ForegroundColor Yellow
    }
}

# OpenMP runtime fallback.
$OmpDll = "$Vs\VC\Redist\MSVC\14.44.35112\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll"
$OmpSource = "Visual Studio BuildTools"
if (-not (Test-Path $OmpDll)) {
    $OmpDll = Join-Path $Root "backends\bin\rocm\libomp140.x86_64.dll"
    $OmpSource = "prebuilt rocm"
}
if (Test-Path $OmpDll) {
    Copy-Item $OmpDll $TargetDir -Force
    Write-Host "  [OMP] $OmpDll -> $TargetDir\libomp140.x86_64.dll ($OmpSource)" -ForegroundColor Yellow
}

# Generate PowerShell launcher.
$LauncherPathLines = ($PathRoots | ForEach-Object { "    '$_'" }) -join ",`n"
@"
`$ROCM_PATHS = @(
$LauncherPathLines
)
`$TargetDir = Split-Path -Parent `$MyInvocation.MyCommand.Path
`$env:Path = "`$(`$ROCM_PATHS -join ';');`$env:SystemRoot\System32\downlevel;`$TargetDir;`$env:Path"

if (`$args.Count -eq 0) {
    Get-ChildItem `$TargetDir -Filter '*.exe' | ForEach-Object { Write-Host `$_.Name }
    exit 0
}
`$cmd = `$args[0]
`$cmdArgs = if (`$args.Count -gt 1) { `$args[1..(`$args.Count - 1)] } else { @() }
`$exe = Join-Path `$TargetDir `$cmd
if (-not (Test-Path `$exe)) { Write-Error "Executable not found: `$exe"; exit 1 }
& `$exe `$cmdArgs
exit `$LASTEXITCODE
"@ | Out-File (Join-Path $TargetDir "run.ps1") -Encoding utf8

# Generate CMD launcher. Percent signs must remain single in the generated file.
@"
@echo off
set "TARGET_DIR=%~dp0"
set "PATH=$($PathRoots -join ';');%SystemRoot%\System32\downlevel;%TARGET_DIR%;%PATH%"
if "%~1"=="" (
    dir "%TARGET_DIR%\*.exe" /b
    exit /b 0
)
set "CMD=%~1"
shift
"%TARGET_DIR%%CMD%" %*
"@ | Out-File (Join-Path $TargetDir "run.bat") -Encoding ascii

Write-Host "Build complete: $TargetDir" -ForegroundColor Green
Write-Host "Direct test: $TargetDir\llama-bench.exe --list-devices" -ForegroundColor Cyan
