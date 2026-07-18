#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup script for llama.cpp inference launcher.
    Downloads binaries, sets up directories, optionally downloads models.
.DESCRIPTION
    Downloads llama.cpp binaries (Vulkan and/or ROCm) from the latest GitHub release,
    sets up the required directory structure, and optionally downloads models.
.PARAMETER VulkanOnly
    Skip ROCm/HIP binary download (Vulkan is enough for most users).
.PARAMETER DownloadModels
    Comma-separated list of models to download. Options: bonsai-27b, bonsai-8b, gemma, qwen35, qwen-coder.
    Or "all" to download everything. Requires HuggingFace token for 27B models.
.PARAMETER HfToken
    HuggingFace token (required for 27B models while repos are gated).
.PARAMETER Version
    llama.cpp release version to download (default: latest b10068).
#>

param(
    [switch]$VulkanOnly,
    [string]$DownloadModels = "",
    [string]$HfToken = "",
    [string]$Version = "b10068"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistDir = Join-Path $ScriptDir "llama.cpp"
$BinDir = Join-Path $DistDir "bin"
$VulkanDir = Join-Path $BinDir "vulkan"
$RocmDir = Join-Path $BinDir "rocm"
$PrismDir = Join-Path $BinDir "prism-hip"
$TemplatesDir = Join-Path $DistDir "templates"
$ModelsDir = Join-Path $DistDir "models"

$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────
function Write-Step {
    param([string]$Msg)
    Write-Host "→ $Msg" -ForegroundColor Cyan
}

function Write-OK {
    Write-Host "  ✓ OK" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Msg)
    Write-Host "  ⚠ $Msg" -ForegroundColor Yellow
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# ──────────────────────────────────────────────────────────────────────
# PREREQUISITES
# ──────────────────────────────────────────────────────────────────────
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  llama.cpp Launcher — Setup" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

Write-Step "Checking prerequisites..."

$hasPython = Test-Command python
if (-not $hasPython) {
    Write-Warn "Python not found. Install Python 3.11+ from https://python.org"
} else {
    $pyVer = python --version 2>&1
    Write-OK
    Write-Host "  Found: $pyVer" -ForegroundColor DarkGray
}

$hasCurl = Test-Command curl
if (-not $hasCurl) {
    Write-Warn "curl not found. Install via: winget install curl"
    Write-Warn "Falling back to .NET WebClient..."
}
$hasGit = Test-Command git
if (-not $hasGit) {
    Write-Host "  git: not required (for setup only)" -ForegroundColor DarkGray
}

# ──────────────────────────────────────────────────────────────────────
# DIRECTORY STRUCTURE
# ──────────────────────────────────────────────────────────────────────
Write-Step "Creating directory structure..."
@($VulkanDir, $RocmDir, $PrismDir, $TemplatesDir, $ModelsDir) | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}
Write-OK

# ──────────────────────────────────────────────────────────────────────
# COPY TEMPLATES
# ──────────────────────────────────────────────────────────────────────
Write-Step "Copying chat templates..."
if (Test-Path (Join-Path $ScriptDir "llama.cpp/templates/chat_template.jinja")) {
    Write-OK
    Write-Host "  Templates already present at llama.cpp/templates/" -ForegroundColor DarkGray
} else {
    Write-Warn "Templates not found. They should be in the repo at llama.cpp/templates/"
}

# ──────────────────────────────────────────────────────────────────────
# DOWNLOAD BINARIES
# ──────────────────────────────────────────────────────────────────────
$repo = "ggml-org/llama.cpp"
$baseUrl = "https://github.com/$repo/releases/download/$Version"

function Download-Binaries {
    param(
        [string]$AssetName,
        [string]$OutputDir,
        [string]$Label
    )
    $url = "$baseUrl/$AssetName"
    $zipPath = Join-Path $env:TEMP "llama-$Label.zip"
    Write-Step "Downloading $Label binaries ($AssetName)..."
    
    if ($hasCurl) {
        curl -L -o "$zipPath" "$url" 2>&1 | Out-Null
    } else {
        (New-Object System.Net.WebClient).DownloadFile($url, $zipPath)
    }
    
    if (-not (Test-Path $zipPath)) {
        Write-Warn "Download failed for $Label"
        return $false
    }
    
    Write-Step "Extracting to $OutputDir..."
    Expand-Archive -Path $zipPath -DestinationPath $OutputDir -Force
    Remove-Item $zipPath -Force
    Write-OK
    return $true
}

# Vulkan
Download-Binaries -AssetName "llama-$Version-bin-win-vulkan-x64.zip" -OutputDir $VulkanDir -Label "Vulkan"

# ROCm (unless VulkanOnly)
if (-not $VulkanOnly) {
    Download-Binaries -AssetName "llama-$Version-bin-win-hip-radeon-x64.zip" -OutputDir $RocmDir -Label "ROCm"
}

# PrismML fork (for Bonsai Q2_0 ROCm support)
$PrismRelease = "prism-b9594-38c66ad"
$PrismUrl = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$PrismRelease/llama-bin-win-hip-radeon-x64.zip"
$prismZip = Join-Path $env:TEMP "llama-prism-hip.zip"

Write-Step "Downloading PrismML ROCm fork (Q2_0 kernels for Bonsai)..."
if ($hasCurl) {
    curl -L -o "$prismZip" "$PrismUrl" 2>&1 | Out-Null
} else {
    (New-Object System.Net.WebClient).DownloadFile($PrismUrl, $prismZip)
}
if (Test-Path $prismZip) {
    Expand-Archive -Path $prismZip -DestinationPath $PrismDir -Force
    Remove-Item $prismZip -Force
    Write-OK
} else {
    Write-Warn "PrismML download failed — Bonsai on ROCm won't work. Vulkan is still fine."
}

# ──────────────────────────────────────────────────────────────────────
# VERIFY BINARIES
# ──────────────────────────────────────────────────────────────────────
Write-Step "Verifying binaries..."
$vulkanExe = Join-Path $VulkanDir "llama-server.exe"
$rocmExe = Join-Path $RocmDir "llama-server.exe"
$prismExe = Join-Path $PrismDir "llama-server.exe"

if (Test-Path $vulkanExe) {
    Write-Host "  ✓ Vulkan:    $vulkanExe" -ForegroundColor Green
} else {
    Write-Warn "Vulkan binary missing at $vulkanExe"
}

if (Test-Path $rocmExe) {
    Write-Host "  ✓ ROCm:      $rocmExe" -ForegroundColor Green
} else {
    Write-Host "  - ROCm:      skipped (use -VulkanOnly or remove flag)" -ForegroundColor DarkGray
}

if (Test-Path $prismExe) {
    Write-Host "  ✓ PrismML:   $prismExe" -ForegroundColor Green
} else {
    Write-Warn "PrismML binary missing — Bonsai on ROCm won't work"
}

# ──────────────────────────────────────────────────────────────────────
# DOWNLOAD MODELS (OPTIONAL)
# ──────────────────────────────────────────────────────────────────────
if ($DownloadModels) {
    Write-Step "Downloading models..."
    
    $models = @{}
    $models["bonsai-27b"] = @{
        Repo = "prism-ml/Ternary-Bonsai-27B-gguf"
        File = "Ternary-Bonsai-27B-Q2_g64.gguf"
        Dir = "prism-ml/Ternary-Bonsai-27B-gguf"
        Token = $true
    }
    $models["bonsai-8b"] = @{
        Repo = "prism-ml/Bonsai-8B-gguf"
        File = "Bonsai-8B-Q1_0.gguf"
        Dir = "prism-ml/Bonsai-8B-gguf"
        Token = $false
    }
    $models["gemma"] = @{
        Repo = "unsloth/gemma-4-12b-it-GGUF"
        File = "gemma-4-12b-it-Q4_K_M.gguf"
        Dir = "unsloth/gemma-4-12b-it-GGUF"
        Token = $false
    }
    $models["qwen35"] = @{
        Repo = "lmstudio-community/Qwen3.5-9B-GGUF"
        File = "Qwen3.5-9B-Q4_K_M.gguf"
        Dir = "lmstudio-community/Qwen3.5-9B-GGUF"
        Token = $false
    }
    $models["qwen-coder"] = @{
        Repo = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
        File = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
        Dir = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
        Token = $false
    }

    $selected = @()
    if ($DownloadModels -eq "all") {
        $selected = $models.Keys
    } else {
        $selected = $DownloadModels.Split(",") | ForEach-Object { $_.Trim().ToLower() }
    }

    foreach ($key in $selected) {
        $m = $models[$key]
        if (-not $m) {
            Write-Warn "Unknown model: $key (skip)"
            continue
        }

        $modelDir = Join-Path $env:USERPROFILE ".lmstudio/models/$($m.Dir)"
        $modelPath = Join-Path $modelDir $m.File

        if (Test-Path $modelPath) {
            Write-Host "  ✓ $key — already downloaded" -ForegroundColor Green
            continue
        }

        if ($m.Token -and -not $HfToken) {
            Write-Warn "Skipping $key — requires HuggingFace token. Use -HfToken parameter"
            continue
        }

        Write-Step "Downloading $key ($($m.File))..."
        New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

        $tokenArg = if ($HfToken) { "--token", $HfToken } else { @() }
        if ($hasGit) {
            & huggingface-cli download $m.Repo $m.File --local-dir $modelDir @tokenArg 2>&1 | Out-Null
            if (Test-Path $modelPath) {
                Write-OK
            } else {
                Write-Warn "Download failed for $key"
            }
        } else {
            Write-Warn "git not found — can't download models automatically"
        }
    }
}

# ──────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================`n" -ForegroundColor Cyan

Write-Host "  Run the launcher:" -ForegroundColor White
Write-Host "    python launcher.py`n" -ForegroundColor Yellow

Write-Host "  Quick start:" -ForegroundColor White
Write-Host "    1. Pick a model from the dropdown" -ForegroundColor DarkGray
Write-Host "    2. Select Vulkan (zero setup) or ROCm (needs rocm-sdk)" -ForegroundColor DarkGray
Write-Host "    3. Click Launch Server, open http://localhost:8081" -ForegroundColor DarkGray
Write-Host "`n  Optional:" -ForegroundColor White
Write-Host "    pip install rocm-sdk    # for ROCm backend" -ForegroundColor DarkGray
Write-Host "`n  Binary locations:" -ForegroundColor DarkGray
Write-Host "    Vulkan:   $VulkanDir" -ForegroundColor DarkGray
Write-Host "    ROCm:     $RocmDir" -ForegroundColor DarkGray
Write-Host "    PrismML:  $PrismDir  (for Bonsai Q2_0 on ROCm)" -ForegroundColor DarkGray
Write-Host "`n"
