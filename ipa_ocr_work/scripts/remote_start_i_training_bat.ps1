$ErrorActionPreference = "Stop"

$Root = "I:\shaoxing_tts_remote\work"
$RunRoot = Join-Path $Root "remote_runs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Python = "C:\Program Files\Python312\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python 3.12 executable not found"
}

$Bat = Join-Path $RunRoot "run_matched_weak.bat"
$Log = Join-Path $RunRoot "matched_weak_train.log"
$Err = Join-Path $RunRoot "matched_weak_train.err.log"
$OutDir = Join-Path $RunRoot "crnn_no_both_matched_weak_e120"

$BatText = @"
@echo on
cd /d "$Root"
echo START %DATE% %TIME%
"$Python" "$Root\train_crnn_ipa_digits.py" --eval-dir "$Root\shaoxing_ipa_no_both_tone" --out-dir "$OutDir" --variant original_export --train-variants original_export --epochs 120 --batch-size 48
echo EXITCODE %ERRORLEVEL% %DATE% %TIME%
"@
Set-Content -LiteralPath $Bat -Value $BatText -Encoding ASCII

$Process = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$Bat`"") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $Log `
    -RedirectStandardError $Err `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath (Join-Path $RunRoot "matched_weak_pid.txt") -Encoding ASCII
Write-Host "PID=$($Process.Id)"
Write-Host "BAT=$Bat"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
