$ErrorActionPreference = "Continue"

$RunRoot = "I:\shaoxing_tts_remote\work\remote_runs"
Write-Host "RUN_ROOT=$RunRoot"
Get-ChildItem -LiteralPath $RunRoot -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize

$PidFile = Join-Path $RunRoot "promoted_then_strict_pid.txt"
if (Test-Path $PidFile) {
    $PidValue = [int](Get-Content -LiteralPath $PidFile)
    Write-Host "PID=$PidValue"
    Get-Process -Id $PidValue -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-Table -AutoSize
}

Write-Host "LOG_TAIL"
$Log = Join-Path $RunRoot "promoted_then_strict_b192.log"
if (Test-Path $Log) {
    Get-Content -LiteralPath $Log -Tail 30
}
Write-Host "ERR_TAIL"
$Err = Join-Path $RunRoot "promoted_then_strict_b192.err.log"
if (Test-Path $Err) {
    Get-Content -LiteralPath $Err -Tail 30
}

Write-Host "PROMOTED_HISTORY"
$History = Join-Path $RunRoot "crnn_promoted_low_cer02_b192_e160\history.tsv"
if (Test-Path $History) {
    Get-Content -LiteralPath $History -Tail 8
}

Write-Host "STRICT_HISTORY"
$StrictHistory = Join-Path $RunRoot "crnn_no_both_strict_b192_e160\history.tsv"
if (Test-Path $StrictHistory) {
    Get-Content -LiteralPath $StrictHistory -Tail 8
}

Write-Host "NVIDIA_SMI"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

Write-Host "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*train_crnn*" -or $_.CommandLine -like "*run_promoted*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    Format-List
