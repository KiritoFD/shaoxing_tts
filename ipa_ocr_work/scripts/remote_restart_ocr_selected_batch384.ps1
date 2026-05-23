$ErrorActionPreference = "Continue"

$RunRoot = "I:\shaoxing_tts_remote\work\remote_runs"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*crnn_ocr_selected_b192_e220*" -or $_.CommandLine -like "*run_ocr_selected_b192_e220*" } |
    ForEach-Object {
        Write-Host "stopping pid=$($_.ProcessId) $($_.Name)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

$ErrorActionPreference = "Stop"
$Root = "I:\shaoxing_tts_remote\work"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Python = "C:\Program Files\Python312\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python 3.12 executable not found"
}

$Log = Join-Path $RunRoot "ocr_selected_b384_e220.log"
$Err = Join-Path $RunRoot "ocr_selected_b384_e220.err.log"
$ScriptPath = Join-Path $RunRoot "run_ocr_selected_b384_e220.bat"
$Train = Join-Path $Root "train_crnn_ipa_digits.py"
$Data = Join-Path $Root "shaoxing_dual_model\ocr_selected"
$Out = Join-Path $RunRoot "crnn_ocr_selected_b384_e220"

$Bat = @"
@echo on
cd /d "$Root"
echo START_OCR_SELECTED_B384
"$Python" "$Train" --eval-dir "$Data" --out-dir "$Out" --variant original_export --train-variants original_export --epochs 220 --batch-size 384 --save-every 30
echo OCR_SELECTED_B384_EXIT %ERRORLEVEL%
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
$Result.ProcessId | Set-Content -LiteralPath (Join-Path $RunRoot "ocr_selected_b384_pid.txt") -Encoding ASCII
Write-Host "PID=$($Result.ProcessId)"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
Write-Host "OUT=$Out"
