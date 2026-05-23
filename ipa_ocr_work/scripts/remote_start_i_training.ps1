$ErrorActionPreference = "Stop"

$Root = "I:\shaoxing_tts_remote"
$Work = Join-Path $Root "work"
$RunRoot = Join-Path $Work "remote_runs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Script = Join-Path $Work "train_crnn_ipa_digits.py"
$EvalDir = Join-Path $Work "shaoxing_ipa_no_both_tone"
$OutDir = Join-Path $RunRoot "crnn_no_both_matched_weak_e120"
$Log = Join-Path $RunRoot "matched_weak_train.log"
$Err = Join-Path $RunRoot "matched_weak_train.err.log"

$PythonCandidates = @(
    "C:\Program Files\Python312\python.exe",
    "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
)
$Python = $PythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Python) {
    throw "Python 3.12 executable not found"
}

$Args = @(
    $Script,
    "--eval-dir", $EvalDir,
    "--out-dir", $OutDir,
    "--variant", "original_export",
    "--train-variants", "original_export",
    "--epochs", "120",
    "--batch-size", "48"
)

$Process = Start-Process -FilePath $Python -ArgumentList $Args `
    -WorkingDirectory $Work `
    -RedirectStandardOutput $Log `
    -RedirectStandardError $Err `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath (Join-Path $RunRoot "matched_weak_pid.txt") -Encoding ASCII
Write-Host "PYTHON=$Python"
Write-Host "PID=$($Process.Id)"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
