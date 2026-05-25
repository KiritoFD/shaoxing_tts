$ErrorActionPreference = "Stop"

$Root = "I:\shaoxing_tts_remote\work_latest"
if (-not (Test-Path $Root)) {
    $Root = "C:\Users\Administrator\shaoxing_tts_remote\work_latest"
}
$RunRoot = Join-Path $Root "remote_runs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Python = "C:\Program Files\Python312\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python 3.12 executable not found"
}

$Name = "trocr_base_printed_ipa_b4_e8"
$Out = Join-Path $RunRoot $Name
$Log = Join-Path $RunRoot "$Name.log"
$Err = Join-Path $RunRoot "$Name.err.log"
$Bat = Join-Path $RunRoot "run_$Name.bat"
$Train = Join-Path $Root "ipa_ocr_work\scripts\train_trocr_wupin.py"
$Score = Join-Path $Root "ipa_ocr_work\scripts\score_ocr_experiment.py"
$Data = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected"
$Pred = Join-Path $Out "predictions_original_export.tsv"
$ScorePrefix = Join-Path $Out "score"

$BatText = @"
@echo on
cd /d "$Root"
echo START_TROCR_IPA
"$Python" "$Train" --eval-dir "$Data" --out-dir "$Out" --variant original_export --train-variants original_export --model microsoft/trocr-base-printed --epochs 8 --batch-size 4 --lr 0.00003 --max-label-length 64
echo TROC_TRAIN_EXIT %ERRORLEVEL%
"$Python" "$Score" --eval-manifest "$Data\eval_manifest.tsv" --predictions "$Pred" --out-prefix "$ScorePrefix" --prediction-mode ipa --include-missing
echo TROC_SCORE_EXIT %ERRORLEVEL%
"@
Set-Content -LiteralPath $Bat -Value $BatText -Encoding ASCII

$CommandLine = "cmd.exe /c `"`"$Bat`" > `"$Log`" 2> `"$Err`"`""
$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $CommandLine
    CurrentDirectory = $Root
}
if ($Result.ReturnValue -ne 0) {
    throw "Win32_Process.Create failed ReturnValue=$($Result.ReturnValue)"
}
$Result.ProcessId | Set-Content -LiteralPath (Join-Path $RunRoot "$Name.pid") -Encoding ASCII
Write-Host "PID=$($Result.ProcessId)"
Write-Host "OUT=$Out"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
