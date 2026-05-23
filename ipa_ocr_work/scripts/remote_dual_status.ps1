$ErrorActionPreference = "Continue"

$RunRoot = "I:\shaoxing_tts_remote\work\remote_runs"
Write-Host "RUN_ROOT=$RunRoot"
Get-ChildItem -LiteralPath $RunRoot -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize

$PidFile = Join-Path $RunRoot "dual_models_pid.txt"
if (Test-Path $PidFile) {
    $PidValue = [int](Get-Content -LiteralPath $PidFile)
    Write-Host "PID=$PidValue"
    Get-Process -Id $PidValue -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-Table -AutoSize
}

Write-Host "LOG_TAIL"
$Log = Join-Path $RunRoot "dual_models.log"
if (Test-Path $Log) {
    Get-Content -LiteralPath $Log -Tail 40
}
Write-Host "ERR_TAIL"
$Err = Join-Path $RunRoot "dual_models.err.log"
if (Test-Path $Err) {
    Get-Content -LiteralPath $Err -Tail 20
}

Write-Host "OCR_HISTORY"
$OcrHistory = Join-Path $RunRoot "crnn_dual_selected_b192_e180\history.tsv"
if (Test-Path $OcrHistory) {
    Get-Content -LiteralPath $OcrHistory -Tail 8
}

Write-Host "DETECTOR_HISTORY"
$DetectorHistory = Join-Path $RunRoot "tone_detector_b512_e80\history.tsv"
if (Test-Path $DetectorHistory) {
    Get-Content -LiteralPath $DetectorHistory -Tail 8
}

Write-Host "NVIDIA_SMI"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

Write-Host "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*train_crnn*" -or $_.CommandLine -like "*train_tone_position*" -or $_.CommandLine -like "*run_dual_models*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    Format-List
