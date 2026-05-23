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

$Log = Join-Path $RunRoot "promoted_then_strict_b192.log"
$Err = Join-Path $RunRoot "promoted_then_strict_b192.err.log"
$ScriptPath = Join-Path $RunRoot "run_promoted_then_strict_b192.bat"
$Train = Join-Path $Root "train_crnn_ipa_digits.py"
$Promoted = Join-Path $Root "shaoxing_ipa_trainable_no_both_promoted_low_cer02"
$Strict = Join-Path $Root "shaoxing_ipa_no_both_tone\strict_matched_only"
$PromotedOut = Join-Path $RunRoot "crnn_promoted_low_cer02_b192_e160"
$StrictOut = Join-Path $RunRoot "crnn_no_both_strict_b192_e160"

$Bat = @"
@echo on
cd /d "$Root"
echo START_PROMOTED %DATE% %TIME%
"$Python" "$Train" --eval-dir "$Promoted" --out-dir "$PromotedOut" --variant original_export --train-variants original_export --epochs 160 --batch-size 192
echo PROMOTED_EXIT %ERRORLEVEL% %DATE% %TIME%
echo START_STRICT %DATE% %TIME%
"$Python" "$Train" --eval-dir "$Strict" --out-dir "$StrictOut" --variant original_export --train-variants original_export --epochs 160 --batch-size 192
echo STRICT_EXIT %ERRORLEVEL% %DATE% %TIME%
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
$Result.ProcessId | Set-Content -LiteralPath (Join-Path $RunRoot "promoted_then_strict_pid.txt") -Encoding ASCII
Write-Host "PID=$($Result.ProcessId)"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
Write-Host "SCRIPT=$ScriptPath"
