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

$Log = Join-Path $RunRoot "dual_models.log"
$Err = Join-Path $RunRoot "dual_models.err.log"
$ScriptPath = Join-Path $RunRoot "run_dual_models.bat"

$TrainOcr = Join-Path $Root "train_crnn_ipa_digits.py"
$TrainDetector = Join-Path $Root "train_tone_position_detector.py"
$OcrData = Join-Path $Root "shaoxing_dual_model\ocr_selected"
$DetectorData = Join-Path $Root "shaoxing_dual_model\tone_position_detector"
$OcrOut = Join-Path $RunRoot "crnn_dual_selected_b192_e180"
$DetectorOut = Join-Path $RunRoot "tone_detector_b512_e80"

$Bat = @"
@echo on
cd /d "$Root"
echo START_OCR
"$Python" "$TrainOcr" --eval-dir "$OcrData" --out-dir "$OcrOut" --variant original_export --train-variants original_export --epochs 180 --batch-size 192 --save-every 30
echo OCR_EXIT %ERRORLEVEL%
echo START_DETECTOR
"$Python" "$TrainDetector" --data-dir "$DetectorData" --out-dir "$DetectorOut" --epochs 80 --batch-size 512 --save-every 10
echo DETECTOR_EXIT %ERRORLEVEL%
"@
Set-Content -LiteralPath $ScriptPath -Value $Bat -Encoding ASCII

$CommandLine = "cmd.exe /c `"`"$ScriptPath`" > `"$Log`" 2> `"$Err`"`""
$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $CommandLine
    CurrentDirectory = $Root
}
if ($Result.ReturnValue -ne 0) {
    throw "Win32_Process.Create failed ReturnValue=$($Result.ReturnValue)"
}
$Result.ProcessId | Set-Content -LiteralPath (Join-Path $RunRoot "dual_models_pid.txt") -Encoding ASCII
Write-Host "PID=$($Result.ProcessId)"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
Write-Host "SCRIPT=$ScriptPath"
