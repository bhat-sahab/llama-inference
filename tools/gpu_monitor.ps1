$exe = if ($env:BENCH_EXE) { $env:BENCH_EXE } else { "D:/Work/llama/backends/bin/rocm/llama-bench.exe" }
$args = @(
    "-m", "C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    "-p", "2048", "-n", "200", "-ngl", "99", "-t", "8",
    "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "1", "-b", "2048", "-ub", "2048",
    "-lm", "none", "--n-cpu-moe", "16"
)
$log = "D:/Work/llama/gpu_monitor.log"
$p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
Write-Host "Started PID $($p.Id) at $(Get-Date -Format HH:mm:ss)"
$deadline = (Get-Date).AddSeconds(240)
$rows = @()
while (-not $p.HasExited -and (Get-Date) -lt $deadline) {
    try {
        $u = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1).CounterSamples |
            Where-Object { $_.CookedValue -gt 1 } | Sort-Object CookedValue -Descending |
            Select-Object -First 4 @{n='eng';e={$_.InstanceName}}, @{n='util';e={[math]::Round($_.CookedValue,1)}}
        $mem = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -SampleInterval 1 -MaxSamples 1).CounterSamples |
            Where-Object { $_.InstanceName -match "pid_$($p.Id)" } |
            ForEach-Object { [math]::Round($_.CookedValue/1MB,0) }
        $proc = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
        $cpu = if ($proc) { [math]::Round($proc.CPU,1) } else { -1 }
        $line = "$(Get-Date -Format HH:mm:ss) | CPU_s=$cpu | VRAM_MB=$($mem -join ',') | " + (($u | ForEach-Object { "$($_.eng)=$($_.util)%" }) -join ' ')
        Write-Host $line
        $rows += $line
    } catch { Write-Host "sample err: $_" }
}
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; Write-Host "TIMEOUT - killed" }
else { Write-Host "EXITED code $($p.ExitCode) at $(Get-Date -Format HH:mm:ss)" }
Write-Host "--- stdout tail ---"
Get-Content $log -Tail 12 -ErrorAction SilentlyContinue
Write-Host "--- stderr tail ---"
Get-Content "$log.err" -Tail 12 -ErrorAction SilentlyContinue
