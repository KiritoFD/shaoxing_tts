$ErrorActionPreference = "Continue"

$Root = "I:\shaoxing_tts_remote\work"
$RunRoot = Join-Path $Root "remote_runs"

Write-Host "RUN_ROOT=$RunRoot"
Get-ChildItem -LiteralPath $RunRoot -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize

$PidPath = Join-Path $RunRoot "matched_weak_pid.txt"
if (Test-Path $PidPath) {
    $PidValue = [int](Get-Content -LiteralPath $PidPath)
    Write-Host "PID=$PidValue"
    Get-Process -Id $PidValue -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-Table -AutoSize
}

Write-Host "STDOUT_TAIL"
$Log = Join-Path $RunRoot "matched_weak_train.log"
if (Test-Path $Log) {
    Get-Content -LiteralPath $Log -Tail 20
}

Write-Host "STDERR_TAIL"
$Err = Join-Path $RunRoot "matched_weak_train.err.log"
if (Test-Path $Err) {
    Get-Content -LiteralPath $Err -Tail 20
}

Write-Host "NVIDIA_SMI"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

Write-Host "HISTORY_TAIL"
$History = Join-Path $RunRoot "crnn_no_both_matched_weak_e120\history.tsv"
if (Test-Path $History) {
    Get-Content -LiteralPath $History -Tail 10
}

Write-Host "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*train_crnn*" -or $_.CommandLine -like "*shaoxing_tts_remote*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    Format-List
