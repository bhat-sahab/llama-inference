# Map GPU engine LUIDs to adapter names, then show utilization per engine
$adapters = Get-CimInstance Win32_VideoController | Select-Object Name, PNPDeviceID, AdapterRAM
$adapters | Format-Table Name, @{n='VRAM_GB';e={[math]::Round($_.AdapterRAM/1GB,1)}} -AutoSize | Out-String | Write-Host

Write-Host "--- llama processes ---"
Get-Process | Where-Object { $_.ProcessName -match 'llama' } | Select-Object Id, ProcessName, CPU, WorkingSet64 | Format-Table -AutoSize | Out-String | Write-Host

Write-Host "--- GPU engine util (3s) ---"
$samples = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 3
$seen = @{}
foreach ($s in $samples) {
    foreach ($c in $s.CounterSamples) {
        if ($c.CookedValue -gt 1) {
            $key = $c.InstanceName
            if (-not $seen.ContainsKey($key)) { $seen[$key] = @() }
            $seen[$key] += [math]::Round($c.CookedValue, 1)
        }
    }
}
foreach ($k in $seen.Keys) { Write-Host "$k  ->  $($seen[$k] -join ', ')" }
