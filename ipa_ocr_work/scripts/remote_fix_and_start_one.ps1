$ErrorActionPreference = "Stop"

$Root = "C:\Users\Administrator\shaoxing_tts_remote"
Set-Location $Root

Get-Process -Id 13852 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

if (Test-Path "ipa_ocr_work") {
    Remove-Item -LiteralPath "ipa_ocr_work" -Recurse -Force
}
Expand-Archive -LiteralPath "remote_training_bundle.zip" -DestinationPath "." -Force

$Script = Join-Path $Root "train_crnn_ipa_digits.py"
if (-not (Test-Path $Script)) {
    throw "train_crnn_ipa_digits.py not found after unzip"
}
$Base = $Root

$EvalDir = Join-Path $Base "shaoxing_ipa_no_both_tone"
if (-not (Test-Path (Join-Path $EvalDir "eval_manifest.tsv"))) {
    throw "eval manifest missing: $EvalDir"
}

$RunRoot = Join-Path $Base "remote_runs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$OutDir = Join-Path $RunRoot "crnn_no_both_matched_weak_e120"
$Log = Join-Path $RunRoot "matched_weak_train.log"
$Err = Join-Path $RunRoot "matched_weak_train.err.log"

$Args = @(
    $Script,
    "--eval-dir", $EvalDir,
    "--out-dir", $OutDir,
    "--variant", "original_export",
    "--train-variants", "original_export",
    "--epochs", "120",
    "--batch-size", "48"
)

$Process = Start-Process -FilePath "py" -ArgumentList (@("-3.12") + $Args) `
    -WorkingDirectory $Base `
    -RedirectStandardOutput $Log `
    -RedirectStandardError $Err `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath (Join-Path $RunRoot "matched_weak_pid.txt") -Encoding ASCII

Write-Host "BASE=$Base"
Write-Host "SCRIPT=$Script"
Write-Host "EVAL=$EvalDir"
Write-Host "OUT=$OutDir"
Write-Host "PID=$($Process.Id)"
Write-Host "LOG=$Log"
Write-Host "ERR=$Err"
