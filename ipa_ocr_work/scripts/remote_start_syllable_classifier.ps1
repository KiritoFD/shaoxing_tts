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

$Name = "syllable_classifier_b160_e120"
$Out = Join-Path $RunRoot $Name
$Log = Join-Path $RunRoot "$Name.log"
$Err = Join-Path $RunRoot "$Name.err.log"
$Bat = Join-Path $RunRoot "run_$Name.bat"
$Train = Join-Path $Root "ipa_ocr_work\scripts\train_syllable_classifier.py"
$Data = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_syllable_ocr"

$BatText = @"
@echo on
cd /d "$Root"
echo START_SYLLABLE_CLASSIFIER
"$Python" "$Train" --eval-dir "$Data" --out-dir "$Out" --variant syllable_crop --epochs 120 --batch-size 160 --height 64 --width 160 --lr 0.001 --save-every 20
echo SYLLABLE_CLASSIFIER_EXIT %ERRORLEVEL%
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
