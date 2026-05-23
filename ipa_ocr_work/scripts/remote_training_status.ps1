$ErrorActionPreference = "Continue"

$Root = "C:\Users\Administrator\shaoxing_tts_remote"
$RunRoot = Join-Path $Root "ipa_ocr_work\models\remote_runs"

Write-Host "RUN_ROOT=$RunRoot"
if (Test-Path $RunRoot) {
    Get-ChildItem -LiteralPath $RunRoot -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
}

$PidPath = Join-Path $RunRoot "pid.txt"
if (Test-Path $PidPath) {
    $PidValue = [int](Get-Content -LiteralPath $PidPath)
Write-Host "PID=$PidValue"
Get-Process -Id $PidValue -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-Table -AutoSize
} else {
    Write-Host "PID file missing"
}

$Log = Join-Path $RunRoot "sequential_train.log"
$Err = Join-Path $RunRoot "sequential_train.err.log"
Write-Host "STDOUT_TAIL"
if (Test-Path $Log) {
    Get-Content -LiteralPath $Log -Tail 20
}
Write-Host "STDERR_TAIL"
if (Test-Path $Err) {
    Get-Content -LiteralPath $Err -Tail 20
}

Write-Host "NVIDIA_SMI"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

Write-Host "TRAINING_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*train_crnn*" -or $_.CommandLine -like "*run_sequential*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    Format-List
