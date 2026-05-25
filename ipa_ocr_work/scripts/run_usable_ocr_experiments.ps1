param(
    [string]$Root = "",
    [string]$RunName = "usable_ocr_path",
    [int]$Batch = 192,
    [double]$MaxHours = 8.0,
    [switch]$SkipTrOCR,
    [switch]$SkipRowCtc,
    [switch]$SkipSyllable,
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"

if (-not $Root) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
if (-not (Test-Path $Root)) {
    throw "Root not found: $Root"
}

$PythonCandidates = @(
    "C:\Program Files\Python312\python.exe",
    "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe",
    (Join-Path $Root ".venv-ocr\Scripts\python.exe"),
    "py -3.12",
    "python"
)

function Resolve-Python {
    foreach ($candidate in $PythonCandidates) {
        if ($candidate -notmatch "\s" -and $candidate -ne "python" -and $candidate -ne "py" -and -not (Test-Path $candidate)) {
            continue
        }
        $probeCommand = if ($candidate -eq "py -3.12") { "py -3.12 -c `"import sys; print(sys.version)`"" } else { "`"$candidate`" -c `"import sys; print(sys.version)`"" }
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        cmd.exe /c $probeCommand >$null 2>$null
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $oldPreference
        if ($exitCode -eq 0) {
            return $candidate
        }
    }
    throw "No Python executable found"
}

$Python = Resolve-Python
$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($MaxHours)
$RunRoot = Join-Path $Root "ipa_ocr_work\runs\$RunName"
$LogRoot = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot | Out-Null

$env:KMP_AFFINITY = "disabled"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$StatusPath = Join-Path $RunRoot "status.json"
$Experiments = @()

function Write-Status([string]$Stage) {
    $gpu = "nvidia-smi unavailable"
    try {
        $gpu = (& nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader 2>$null) -join "`n"
    } catch {}
    $payload = [ordered]@{
        stage = $Stage
        root = $Root
        run_root = $RunRoot
        python = $Python
        started_at = $StartedAt.ToString("s")
        deadline = $Deadline.ToString("s")
        updated_at = (Get-Date).ToString("s")
        gpu = $gpu
        experiments = $Experiments
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Add-Experiment([string]$Name, [string]$State, [string]$Reason, [string]$OutDir) {
    $script:Experiments += [ordered]@{
        name = $Name
        state = $State
        reason = $Reason
        out_dir = $OutDir
        updated_at = (Get-Date).ToString("s")
    }
    Write-Status $Name
}

function Update-Experiment([string]$Name, [string]$State, [string]$Reason) {
    for ($i = 0; $i -lt $script:Experiments.Count; $i++) {
        if ($script:Experiments[$i].name -eq $Name) {
            $script:Experiments[$i].state = $State
            $script:Experiments[$i].reason = $Reason
            $script:Experiments[$i].updated_at = (Get-Date).ToString("s")
        }
    }
    Write-Status $Name
}

function Run-Command([string]$Name, [string[]]$ArgList, [string]$OutDir) {
    if ((Get-Date) -ge $Deadline) {
        Add-Experiment $Name "skipped" "deadline reached before start" $OutDir
        return 124
    }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Add-Experiment $Name "running" "" $OutDir
    $log = Join-Path $LogRoot "$Name.log"
    $err = Join-Path $LogRoot "$Name.err.log"
    if ($Python -eq "py -3.12") {
        $proc = Start-Process -FilePath "py" -ArgumentList (@("-3.12") + $ArgList) -WorkingDirectory $Root -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
    } else {
        $proc = Start-Process -FilePath $Python -ArgumentList $ArgList -WorkingDirectory $Root -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
    }
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 30
        $proc.Refresh()
        Write-Status $Name
        if ((Get-Date) -ge $Deadline) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Update-Experiment $Name "timeout" "deadline reached"
            return 124
        }
    }
    $proc.WaitForExit()
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) {
        $errItem = Get-Item $err -ErrorAction SilentlyContinue
        if ($errItem -and $errItem.Length -eq 0) {
            $exitCode = 0
        }
    }
    if ($exitCode -eq 0) {
        Update-Experiment $Name "done" ""
    } else {
        Update-Experiment $Name "failed" "exit code $exitCode; see $log and $err"
    }
    return $exitCode
}

function Score-RowPredictions([string]$Name, [string]$Predictions, [string]$OutDir, [string]$Mode) {
    $manifest = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected\eval_manifest.tsv"
    $prefix = Join-Path $OutDir "score"
    $args = @(
        "ipa_ocr_work\scripts\score_ocr_experiment.py",
        "--eval-manifest", $manifest,
        "--predictions", $Predictions,
        "--out-prefix", $prefix,
        "--prediction-mode", $Mode,
        "--ipa-label-source", "from-wupin",
        "--include-missing"
    )
    return Run-Command "${Name}_score" $args $OutDir
}

function Decode-And-Score([string]$Name, [string]$Predictions, [string]$OutDir, [string]$Mode) {
    $manifest = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected\eval_manifest.tsv"
    $decoded = Join-Path $OutDir "predictions_lexicon.tsv"
    $decodeArgs = @(
        "ipa_ocr_work\scripts\apply_wupin_lexicon_decoder.py",
        "--eval-manifest", $manifest,
        "--predictions", $Predictions,
        "--out", $decoded,
        "--prediction-mode", $Mode,
        "--lexicon-split", "train",
        "--max-syllable-distance", "2",
        "--row-nearest",
        "--row-nearest-max-cer", "0.18"
    )
    $code = Run-Command "${Name}_lexicon_decode" $decodeArgs $OutDir
    if ($code -ne 0) { return $code }
    return Score-RowPredictions "${Name}_lexicon" $decoded $OutDir "wupin"
}

Write-Status "initializing"

$rowData = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected"
$syllData = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_syllable_ocr"

if (-not (Test-Path (Join-Path $rowData "eval_manifest.tsv"))) {
    throw "Missing row eval manifest: $rowData"
}
if (-not (Test-Path (Join-Path $syllData "eval_manifest.tsv"))) {
    throw "Missing syllable eval manifest: $syllData"
}

if ($SmokeOnly) {
    $SmokeOnly = $true
    $Batch = [Math]::Min($Batch, 16)
}
$CtcBatch = if ($SmokeOnly) { 16 } else { 192 }

if (-not $SkipSyllable) {
    $out = Join-Path $RunRoot "E1_syllable_closed_set"
    $epochs = if ($SmokeOnly) { 2 } else { 180 }
    $batchLocal = [Math]::Min($Batch, 256)
    $args = @(
        "ipa_ocr_work\scripts\train_syllable_classifier.py",
        "--eval-dir", $syllData,
        "--out-dir", $out,
        "--variant", "syllable_crop",
        "--epochs", "$epochs",
        "--batch-size", "$batchLocal",
        "--height", "64",
        "--width", "192",
        "--lr", "0.001",
        "--save-every", "20"
    )
    $code = Run-Command "E1_syllable_closed_set" $args $out
    if ($code -eq 0) {
        $pred = Join-Path $out "predictions_syllable_crop.tsv"
        $scoreArgs = @(
            "ipa_ocr_work\scripts\score_syllable_ocr_rows.py",
            "--manifest", (Join-Path $syllData "eval_manifest.tsv"),
            "--predictions", $pred,
            "--out", (Join-Path $out "row_score.tsv")
        )
        Run-Command "E1_syllable_closed_set_row_score" $scoreArgs $out | Out-Null
    }
}

if (-not $SkipRowCtc) {
    $out = Join-Path $RunRoot "E2_variable_width_ctc_svtr_tiny"
    $epochs = if ($SmokeOnly) { 2 } else { 220 }
    $args = @(
        "ipa_ocr_work\scripts\train_crnn_ipa_digits.py",
        "--eval-dir", $rowData,
        "--out-dir", $out,
        "--variant", "original_export",
        "--train-variants", "original_export",
        "--epochs", "$epochs",
        "--batch-size", "$CtcBatch",
        "--height", "64",
        "--max-width", "960",
        "--backbone", "svtr_tiny",
        "--lr", "0.0005",
        "--save-every", "20"
    )
    $code = Run-Command "E2_variable_width_ctc_svtr_tiny" $args $out
    if ($code -eq 0) {
        $pred = Join-Path $out "predictions_original_export.tsv"
        Score-RowPredictions "E2_variable_width_ctc_svtr_tiny" $pred $out "ipa" | Out-Null
        Decode-And-Score "E2_variable_width_ctc_svtr_tiny" $pred $out "ipa" | Out-Null
    }
}

if (-not $SkipTrOCR) {
    $out = Join-Path $RunRoot "E3_trocr_pad_square_control"
    $epochs = if ($SmokeOnly) { 1 } else { 4 }
    $args = @(
        "ipa_ocr_work\scripts\train_trocr_wupin.py",
        "--eval-dir", $rowData,
        "--out-dir", $out,
        "--variant", "original_export",
        "--train-variants", "original_export",
        "--model", "microsoft/trocr-base-printed",
        "--epochs", "$epochs",
        "--batch-size", "4",
        "--lr", "0.00001",
        "--max-label-length", "64",
        "--label-source", "ipa-from-wupin",
        "--image-mode", "pad-square"
    )
    $code = Run-Command "E3_trocr_pad_square_control" $args $out
    if ($code -eq 0) {
        $pred = Join-Path $out "predictions_original_export.tsv"
        Score-RowPredictions "E3_trocr_pad_square_control" $pred $out "ipa" | Out-Null
        Decode-And-Score "E3_trocr_pad_square_control" $pred $out "ipa" | Out-Null
    }
}

Write-Status "finished"
Write-Host "STATUS=$StatusPath"
