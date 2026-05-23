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

$Log = Join-Path $RunRoot "tone_detector_remote_b512_e100.log"
$Err = Join-Path $RunRoot "tone_detector_remote_b512_e100.err.log"
$ScriptPath = Join-Path $RunRoot "run_tone_detector_remote_b512_e100.bat"
$Train = Join-Path $Root "train_tone_position_detector.py"
$Data = Join-Path $Root "shaoxing_dual_model\tone_position_detector"
$Out = Join-Path $RunRoot "tone_detector_remote_b512_e100"

$Bat = @"
@echo on
cd /d "$Root"
echo START_DETECTOR
"$Python" "$Train" --data-dir "$Data" --out-dir "$Out" --epochs 100 --batch-size 512 --save-every 10
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
$Result.ProcessId | Set-Content -LiteralPath (Join-Path $RunRoot "tone_detector_remote_pid.txt") -Encoding ASCII
Write-Host "PID=$($Result.ProcessId)"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
Write-Host "OUT=$Out"
