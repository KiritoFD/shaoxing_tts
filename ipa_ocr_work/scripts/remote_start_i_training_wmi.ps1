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

$Log = Join-Path $RunRoot "matched_weak_train.log"
$Err = Join-Path $RunRoot "matched_weak_train.err.log"
$OutDir = Join-Path $RunRoot "crnn_no_both_matched_weak_e120"
$Script = Join-Path $Root "train_crnn_ipa_digits.py"
$EvalDir = Join-Path $Root "shaoxing_ipa_no_both_tone"

$Inner = "`"$Python`" `"$Script`" --eval-dir `"$EvalDir`" --out-dir `"$OutDir`" --variant original_export --train-variants original_export --epochs 120 --batch-size 48 > `"$Log`" 2> `"$Err`""
$CommandLine = "cmd.exe /c `"$Inner`""

$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $CommandLine
    CurrentDirectory = $Root
}
if ($Result.ReturnValue -ne 0) {
    throw "Win32_Process.Create failed ReturnValue=$($Result.ReturnValue)"
}
$Result.ProcessId | Set-Content -LiteralPath (Join-Path $RunRoot "matched_weak_pid.txt") -Encoding ASCII

Write-Host "PID=$($Result.ProcessId)"
Write-Host "CMD=$CommandLine"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
