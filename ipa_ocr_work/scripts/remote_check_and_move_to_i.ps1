$ErrorActionPreference = "Continue"

Write-Host "DRIVES"
Get-CimInstance Win32_LogicalDisk |
    Select-Object DeviceID,FileSystem,@{Name="FreeGB";Expression={[math]::Round($_.FreeSpace / 1GB, 2)}},@{Name="SizeGB";Expression={[math]::Round($_.Size / 1GB, 2)}} |
    Format-Table -AutoSize

Write-Host "CURRENT_ROOT"
$CurrentRoot = "C:\Users\Administrator\shaoxing_tts_remote"
if (Test-Path $CurrentRoot) {
    Get-ChildItem -LiteralPath $CurrentRoot -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
}

Write-Host "CURRENT_REMOTE_RUNS"
$CurrentRuns = Join-Path $CurrentRoot "remote_runs"
if (Test-Path $CurrentRuns) {
    Get-ChildItem -LiteralPath $CurrentRuns -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
    if (Test-Path (Join-Path $CurrentRuns "matched_weak_pid.txt")) {
        $PidValue = [int](Get-Content -LiteralPath (Join-Path $CurrentRuns "matched_weak_pid.txt"))
        Write-Host "CURRENT_PID=$PidValue"
        Get-Process -Id $PidValue -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-Table -AutoSize
    }
}

Write-Host "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*train_crnn*" -or $_.CommandLine -like "*shaoxing_tts_remote*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    Format-List
