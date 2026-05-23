$ErrorActionPreference = "Stop"

$SourceRoot = "C:\Users\Administrator\shaoxing_tts_remote"
$Root = "I:\shaoxing_tts_remote"

New-Item -ItemType Directory -Force -Path $Root | Out-Null
Copy-Item -LiteralPath (Join-Path $SourceRoot "remote_training_bundle.zip") -Destination (Join-Path $Root "remote_training_bundle.zip") -Force
Set-Location $Root

if (Test-Path "work") {
    Remove-Item -LiteralPath "work" -Recurse -Force
}
New-Item -ItemType Directory -Force -Path "work" | Out-Null
Expand-Archive -LiteralPath "remote_training_bundle.zip" -DestinationPath "work" -Force

$Script = Join-Path $Root "work\train_crnn_ipa_digits.py"
$EvalDir = Join-Path $Root "work\shaoxing_ipa_no_both_tone"
$SmokeOut = Join-Path $Root "work\remote_runs\smoke_e1"
New-Item -ItemType Directory -Force -Path (Split-Path $SmokeOut -Parent) | Out-Null

Write-Host "ROOT=$Root"
Write-Host "SCRIPT=$Script"
Write-Host "EVAL=$EvalDir"
py -3.12 $Script --eval-dir $EvalDir --out-dir $SmokeOut --variant original_export --train-variants original_export --epochs 1 --batch-size 8 --eval-limit 16
