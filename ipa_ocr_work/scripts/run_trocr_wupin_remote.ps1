$ErrorActionPreference = "Stop"

$root = "I:\shaoxing_tts_remote\work_latest"
$runRoot = Join-Path $root "remote_runs"
$name = "E3_trocr_base_printed_wupin_pad_square_b4_lr3e5_e12"
$python = "C:\Program Files\Python312\python.exe"
$outDir = Join-Path $runRoot $name
$statusLog = Join-Path $runRoot ($name + ".status.log")
$trainLog = Join-Path $runRoot ($name + ".train.log")
$trainErr = Join-Path $runRoot ($name + ".train.err.log")
$scoreLog = Join-Path $runRoot ($name + ".score.log")
$scoreErr = Join-Path $runRoot ($name + ".score.err.log")

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $root

"START_RUN $(Get-Date -Format o)" | Tee-Object -FilePath $statusLog -Append

$trainArgs = @(
    "-u", "I:\shaoxing_tts_remote\work_latest\ipa_ocr_work\scripts\train_trocr_wupin.py",
    "--eval-dir", "I:\shaoxing_tts_remote\work_latest\ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected",
    "--out-dir", $outDir,
    "--variant", "original_export",
    "--train-variants", "original_export",
    "--model", "microsoft/trocr-base-printed",
    "--epochs", "12",
    "--batch-size", "4",
    "--lr", "0.00003",
    "--max-label-length", "48",
    "--label-source", "wupin",
    "--image-mode", "pad-square"
)

& $python @trainArgs > $trainLog 2> $trainErr
$trainExit = $LASTEXITCODE
"TRAIN_EXIT $trainExit $(Get-Date -Format o)" | Tee-Object -FilePath $statusLog -Append
if ($trainExit -ne 0) {
    exit $trainExit
}

$scoreArgs = @(
    "-u", "I:\shaoxing_tts_remote\work_latest\ipa_ocr_work\scripts\score_ocr_experiment.py",
    "--eval-manifest", "I:\shaoxing_tts_remote\work_latest\ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected\eval_manifest.tsv",
    "--predictions", (Join-Path $outDir "predictions_original_export.tsv"),
    "--out-prefix", (Join-Path $outDir "score"),
    "--prediction-mode", "wupin",
    "--ipa-label-source", "from-wupin",
    "--include-missing"
)

& $python @scoreArgs > $scoreLog 2> $scoreErr
$scoreExit = $LASTEXITCODE
"SCORE_EXIT $scoreExit $(Get-Date -Format o)" | Tee-Object -FilePath $statusLog -Append
exit $scoreExit
