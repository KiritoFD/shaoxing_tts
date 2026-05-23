$ErrorActionPreference = "Continue"

$RunRoot = "I:\shaoxing_tts_remote\work\remote_runs"
$RunDir = Join-Path $RunRoot "crnn_ocr_selected_b324_e220"
Write-Host "RUN_DIR=$RunDir"
if (Test-Path $RunDir) {
    Get-ChildItem -LiteralPath $RunDir -Force | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
}

$PidPath = Join-Path $RunRoot "ocr_selected_b324_pid.txt"
if (Test-Path $PidPath) {
    $PidValue = [int](Get-Content -LiteralPath $PidPath)
    Write-Host "PID=$PidValue"
    Get-Process -Id $PidValue -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime | Format-Table -AutoSize
}

Write-Host "LOG_TAIL"
$Log = Join-Path $RunRoot "ocr_selected_b324_e220.log"
if (Test-Path $Log) {
    Get-Content -LiteralPath $Log -Tail 25
}

Write-Host "ERR_TAIL"
$Err = Join-Path $RunRoot "ocr_selected_b324_e220.err.log"
if (Test-Path $Err) {
    Get-Content -LiteralPath $Err -Tail 15
}

Write-Host "HISTORY_TAIL"
$History = Join-Path $RunDir "history.tsv"
if (Test-Path $History) {
    Get-Content -LiteralPath $History -Tail 12
}

Write-Host "CHECKPOINT_EVAL"
$Eval = Join-Path $RunDir "checkpoint_eval.tsv"
if (Test-Path $Eval) {
    Get-Content -LiteralPath $Eval
}

Write-Host "BEST_FROM_HISTORY"
if (Test-Path $History) {
    $Rows = Import-Csv -LiteralPath $History -Delimiter "`t"
    if ($Rows.Count -gt 0) {
        $BestCer = $Rows | Sort-Object {[double]$_.val_cer} | Select-Object -First 1
        $BestExact = $Rows | Sort-Object {[double]$_.val_exact} -Descending | Select-Object -First 1
        Write-Host "best_val_cer_epoch=$($BestCer.epoch) val_cer=$($BestCer.val_cer) val_exact=$($BestCer.val_exact)"
        Write-Host "best_val_exact_epoch=$($BestExact.epoch) val_exact=$($BestExact.val_exact) val_cer=$($BestExact.val_cer)"
    }
}

Write-Host "NVIDIA_SMI"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

Write-Host "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*crnn_ocr_selected_b324_e220*" -or $_.CommandLine -like "*run_ocr_selected_b324_e220*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine |
    Format-List
