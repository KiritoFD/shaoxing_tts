$ErrorActionPreference = "Stop"

$Root = "C:\Users\Administrator\shaoxing_tts_remote"
Set-Location $Root

if (Test-Path "ipa_ocr_work") {
    Remove-Item -LiteralPath "ipa_ocr_work" -Recurse -Force
}
Expand-Archive -LiteralPath "remote_training_bundle.zip" -DestinationPath "." -Force

$RunRoot = Join-Path $Root "ipa_ocr_work\models\remote_runs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Script = Join-Path $Root "ipa_ocr_work\scripts\train_crnn_ipa_digits.py"
$Log = Join-Path $RunRoot "sequential_train.log"
$Err = Join-Path $RunRoot "sequential_train.err.log"

$Command = @"
`$ErrorActionPreference = "Stop"
Set-Location "$Root"
py -3.12 "$Script" --eval-dir "ipa_ocr_work\dataset\shaoxing_ipa_no_both_tone" --out-dir "ipa_ocr_work\models\remote_runs\crnn_no_both_matched_weak_e120" --variant original_export --train-variants original_export --epochs 120 --batch-size 48
py -3.12 "$Script" --eval-dir "ipa_ocr_work\dataset\shaoxing_ipa_no_both_tone\strict_matched_only" --out-dir "ipa_ocr_work\models\remote_runs\crnn_no_both_strict_e120" --variant original_export --train-variants original_export --epochs 120 --batch-size 48
"@

$JobScript = Join-Path $RunRoot "run_sequential.ps1"
Set-Content -LiteralPath $JobScript -Value $Command -Encoding UTF8

$Process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $JobScript
) -RedirectStandardOutput $Log -RedirectStandardError $Err -WindowStyle Hidden -PassThru

$Process.Id | Set-Content -LiteralPath (Join-Path $RunRoot "pid.txt") -Encoding ASCII
Write-Host "started pid=$($Process.Id)"
Write-Host "log=$Log"
Write-Host "err=$Err"
