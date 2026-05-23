$ErrorActionPreference = "Continue"

$RunRoot = "I:\shaoxing_tts_remote\work\remote_runs"
Write-Host "RUN_ROOT=$RunRoot"
Get-ChildItem -LiteralPath $RunRoot -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize

$PidPath = Join-Path $RunRoot "ocr_selected_pid.txt"
if (Test-Path $PidPath) {
    $PidValue = [int](Get-Content -LiteralPath $PidPath)
    Write-Host "PID=$PidValue"
    Get-Process -Id $PidValue -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-Table -AutoSize
}

Write-Host "LOG_TAIL"
$Log = Join-Path $RunRoot "ocr_selected_b192_e220.log"
if (Test-Path $Log) {
    Get-Content -LiteralPath $Log -Tail 30
}

Write-Host "ERR_TAIL"
$Err = Join-Path $RunRoot "ocr_selected_b192_e220.err.log"
if (Test-Path $Err) {
    Get-Content -LiteralPath $Err -Tail 20
}

Write-Host "HISTORY_TAIL"
$History = Join-Path $RunRoot "crnn_ocr_selected_b192_e220\history.tsv"
if (Test-Path $History) {
    Get-Content -LiteralPath $History -Tail 10
}

Write-Host "CHECKPOINT_EVAL"
$Eval = Join-Path $RunRoot "crnn_ocr_selected_b192_e220\checkpoint_eval.tsv"
if (Test-Path $Eval) {
    Get-Content -LiteralPath $Eval
}

Write-Host "NVIDIA_SMI"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

Write-Host "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*train_crnn_ipa_digits*" -or $_.CommandLine -like "*run_ocr_selected*" -or $_.CommandLine -like "*train_tone_position_detector*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    Format-List
