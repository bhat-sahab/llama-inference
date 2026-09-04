# reset-gpu.ps1 — restart the AMD RX 9070 XT via pnputil /restart-device.
# Fixes GPU stuck at idle clocks (llama tg collapse, e.g. tg4 32 -> 2.9 t/s).
# Requires an elevated PowerShell. Called from run_model.bat (self-elevates).
#
# Safety: the screen may flicker/go black for a few seconds.
# The RX 9070 XT should NOT be the active display adapter (use iGPU/other).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File reset-gpu.ps1
#   powershell -ExecutionPolicy Bypass -File reset-gpu.ps1 -InstanceId "PCI\VEN_1002&DEV_7550&SUBSYS_34901DA2&REV_C0\6&20695DCE&0&00000008"

param(
    [string]$InstanceId = ""
)

$gpu = Get-PnpDevice -Class Display | Where-Object { $_.FriendlyName -like "*9070 XT*" }
if (-not $gpu) {
    Write-Host "ERROR: RX 9070 XT not found among display devices" -ForegroundColor Red
    exit 1
}

if ($InstanceId -ne "") {
    $id = $InstanceId
    Write-Host "Using provided InstanceId: $id" -ForegroundColor Cyan
} else {
    $id = $gpu.InstanceId
}

Write-Host "Resetting: $($gpu.FriendlyName)" -ForegroundColor Cyan
Write-Host "  InstanceId: $id" -ForegroundColor Cyan
Write-Host "Screen may flicker/black for a few seconds..." -ForegroundColor Yellow

# pnputil /restart-device (single disable+enable, cleanest)
& pnputil /restart-device $id
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "pnputil /restart-device failed (code $code) - falling back to PnP disable/enable" -ForegroundColor Yellow
    Disable-PnpDevice -InstanceId $id -Confirm:$false -ErrorAction Stop
    Start-Sleep -Seconds 2
    Enable-PnpDevice -InstanceId $id -Confirm:$false -ErrorAction Stop
    Start-Sleep -Seconds 5
}

Start-Sleep -Seconds 3
$check = Get-PnpDevice -Class Display | Where-Object { $_.InstanceId -eq $id }
if ($check -and $check.Status -eq "OK") {
    Write-Host "GPU reset OK - status: $($check.Status)" -ForegroundColor Green
} else {
    Write-Host "WARNING: GPU status after reset: $($check.Status) (problem: $($check.Problem))" -ForegroundColor Red
    exit 2
}
