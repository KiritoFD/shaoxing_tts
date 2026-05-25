$ErrorActionPreference = "Continue"

$Root = "I:\shaoxing_tts_remote\work_latest"
if (-not (Test-Path $Root)) {
    $Root = "C:\Users\Administrator\shaoxing_tts_remote\work_latest"
}
$RunRoot = Join-Path $Root "remote_runs"
$Name = "syllable_svtr_b160_e180"
$Out = Join-Path $RunRoot $Name
$Log = Join-Path $RunRoot "$Name.log"
$Err = Join-Path $RunRoot "$Name.err.log"

Write-Host "OUT=$Out"
Get-ChildItem -LiteralPath $Out -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime
Write-Host "LOG_TAIL"
Get-Content -LiteralPath $Log -Tail 24 -ErrorAction SilentlyContinue
Write-Host "ERR_TAIL"
Get-Content -LiteralPath $Err -Tail 12 -ErrorAction SilentlyContinue
Write-Host "HISTORY_TAIL"
Get-Content -LiteralPath (Join-Path $Out "history.tsv") -Tail 12 -ErrorAction SilentlyContinue
Write-Host "CHECKPOINT_EVAL"
Get-Content -LiteralPath (Join-Path $Out "checkpoint_eval.tsv") -ErrorAction SilentlyContinue
Write-Host "NVIDIA_SMI"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
Write-Host "TRAIN_PROCESSES"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*syllable_svtr_b160_e180*" -or $_.CommandLine -like "*train_crnn_ipa_digits.py*" } |
    Select-Object ProcessId,ParentProcessId,Name,CommandLine
