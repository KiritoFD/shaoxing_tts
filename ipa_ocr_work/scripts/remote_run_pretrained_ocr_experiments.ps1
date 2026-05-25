$ErrorActionPreference = "Continue"

param(
    [string]$Root = "",
    [int]$Batch = 96,
    [double]$MaxHours = 8.0,
    [switch]$InstallDeps
)

if (-not $Root) {
    $Root = "I:\shaoxing_tts_remote\work_latest"
    if (-not (Test-Path $Root)) {
        $Root = "C:\Users\Administrator\shaoxing_tts_remote\work_latest"
    }
}
if (-not (Test-Path $Root)) {
    throw "Root not found: $Root"
}

$RunRoot = Join-Path $Root "remote_runs\pretrained_ocr_8h"
$LogRoot = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot | Out-Null

$StatusPath = Join-Path $RunRoot "status.json"
$StartedAt = Get-Date
$Deadline = $StartedAt.AddHours($MaxHours)
$Experiments = @()

function Resolve-Python {
    $candidates = @(
        "C:\Program Files\Python312\python.exe",
        "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe",
        "py"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -eq "py") {
            return "py -3.12"
        }
        if (Test-Path $candidate) {
            return "`"$candidate`""
        }
    }
    throw "Python 3.12 executable not found"
}

function Test-PythonPackage([string]$Package) {
    $code = "import importlib.util as u; raise SystemExit(0 if u.find_spec('$Package') else 1)"
    $tmp = Join-Path $env:TEMP "test_pkg_$Package.py"
    Set-Content -LiteralPath $tmp -Value $code -Encoding UTF8
    $python = Resolve-Python
    cmd.exe /c "$python `"$tmp`"" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Write-Status([string]$Stage = "") {
    $gpu = ""
    try {
        $gpu = (& nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>$null) -join "`n"
    } catch {
        $gpu = "nvidia-smi unavailable"
    }
    $active = @()
    try {
        $active = Get-CimInstance Win32_Process |
            Where-Object { $_.CommandLine -like "*pretrained_ocr_8h*" -or $_.CommandLine -like "*train_trocr*" -or $_.CommandLine -like "*train_syllable_classifier*" -or $_.CommandLine -like "*run_ocr_ipa_calamari_row*" } |
            Select-Object ProcessId,Name,CommandLine
    } catch {
        $active = @()
    }
    $payload = [ordered]@{
        stage = $Stage
        root = $Root
        run_root = $RunRoot
        started_at = $StartedAt.ToString("s")
        deadline = $Deadline.ToString("s")
        updated_at = (Get-Date).ToString("s")
        batch_default = $Batch
        gpu = $gpu
        experiments = $Experiments
        active_processes = $active
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

function Update-Experiment([string]$Name, [string]$State, [string]$Reason = "") {
    for ($i = 0; $i -lt $script:Experiments.Count; $i++) {
        if ($script:Experiments[$i].name -eq $Name) {
            $script:Experiments[$i].state = $State
            $script:Experiments[$i].reason = $Reason
            $script:Experiments[$i].updated_at = (Get-Date).ToString("s")
        }
    }
    Write-Status $Name
}

function Run-Step([string]$Name, [string]$Command, [string]$OutDir) {
    if ((Get-Date) -ge $Deadline) {
        Add-Experiment $Name "skipped" "deadline reached before start" $OutDir
        return 124
    }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Add-Experiment $Name "running" "" $OutDir
    $bat = Join-Path $LogRoot "run_$Name.bat"
    $log = Join-Path $LogRoot "$Name.log"
    $err = Join-Path $LogRoot "$Name.err.log"
    $batText = @"
@echo on
cd /d "$Root"
$Command
exit /b %ERRORLEVEL%
"@
    Set-Content -LiteralPath $bat -Value $batText -Encoding ASCII
    $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"`"$bat`" > `"$log`" 2> `"$err`"`"" -WorkingDirectory $Root -PassThru -WindowStyle Hidden
    while (-not $proc.HasExited) {
        Write-Status $Name
        Start-Sleep -Seconds 30
        $proc.Refresh()
        if ((Get-Date) -ge $Deadline) {
            try {
                Stop-Process -Id $proc.Id -Force
            } catch {}
            Update-Experiment $Name "timeout" "deadline reached"
            return 124
        }
    }
    if ($proc.ExitCode -eq 0) {
        Update-Experiment $Name "done" ""
    } else {
        Update-Experiment $Name "failed" "exit code $($proc.ExitCode); see $log and $err"
    }
    return $proc.ExitCode
}

Write-Status "initializing"

try {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*syllable_svtr*" -or $_.CommandLine -like "*train_svtr*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
} catch {}

$Python = Resolve-Python

if ($InstallDeps) {
    $installLog = Join-Path $LogRoot "install_deps.log"
    $installErr = Join-Path $LogRoot "install_deps.err.log"
    $installCmd = @"
$Python -m pip install --upgrade pip
$Python -m pip install "tensorflow>=2.16,<2.19"
$Python -m pip install paddleocr
$Python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
"@
    Add-Experiment "deps" "running" "" $RunRoot
    $depsBat = Join-Path $LogRoot "install_deps.bat"
    Set-Content -LiteralPath $depsBat -Value $installCmd -Encoding ASCII
    $depsProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"`"$depsBat`" > `"$installLog`" 2> `"$installErr`"`"" -WorkingDirectory $Root -PassThru -WindowStyle Hidden
    $depsProc.WaitForExit()
    if ($depsProc.ExitCode -eq 0) {
        Update-Experiment "deps" "done" ""
    } else {
        Update-Experiment "deps" "failed" "exit code $($depsProc.ExitCode); continuing with available packages"
    }
}

$Data = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected"
$Manifest = Join-Path $Data "eval_manifest.tsv"
$Score = Join-Path $Root "ipa_ocr_work\scripts\score_ocr_experiment.py"

if ((Test-PythonPackage "tensorflow") -and (Test-Path (Join-Path $Root "ocr-ipa-main\model\calamari\best.ckpt"))) {
    $Out = Join-Path $RunRoot "E1_ocr_ipa_calamari_row"
    $Pred = Join-Path $Out "predictions.tsv"
    $ScorePrefix = Join-Path $Out "score"
    $Cmd = @"
$Python "$Root\ipa_ocr_work\scripts\run_ocr_ipa_calamari_row.py" --eval-dir "$Data" --out "$Pred"
$Python "$Score" --eval-manifest "$Manifest" --predictions "$Pred" --out-prefix "$ScorePrefix" --prediction-mode ipa --include-missing
"@
    Run-Step "E1_ocr_ipa_calamari_row" $Cmd $Out | Out-Null
} else {
    Add-Experiment "E1_ocr_ipa_calamari_row" "skipped" "tensorflow or ocr-ipa calamari checkpoint unavailable" (Join-Path $RunRoot "E1_ocr_ipa_calamari_row")
}

if ((Test-PythonPackage "paddle") -and (Test-PythonPackage "paddleocr")) {
    Add-Experiment "E2_paddleocr_ppocrv5" "skipped" "Paddle packages are present, but PP-OCRv5 fine-tune entry is not wired in this workspace yet" (Join-Path $RunRoot "E2_paddleocr_ppocrv5")
} else {
    Add-Experiment "E2_paddleocr_ppocrv5" "skipped" "paddle or paddleocr unavailable after dependency preparation" (Join-Path $RunRoot "E2_paddleocr_ppocrv5")
}

if ((Test-PythonPackage "torch") -and (Test-PythonPackage "transformers")) {
    $Out = Join-Path $RunRoot "E3_trocr_base_printed_ipa"
    $Pred = Join-Path $Out "predictions_original_export.tsv"
    $ScorePrefix = Join-Path $Out "score"
    $Cmd = @"
$Python "$Root\ipa_ocr_work\scripts\train_trocr_wupin.py" --eval-dir "$Data" --out-dir "$Out" --variant original_export --train-variants original_export --model microsoft/trocr-base-printed --epochs 8 --batch-size 4 --lr 0.00003 --max-label-length 80
$Python "$Score" --eval-manifest "$Manifest" --predictions "$Pred" --out-prefix "$ScorePrefix" --prediction-mode ipa --include-missing
"@
    Run-Step "E3_trocr_base_printed_ipa" $Cmd $Out | Out-Null
} else {
    Add-Experiment "E3_trocr_base_printed_ipa" "skipped" "torch or transformers unavailable" (Join-Path $RunRoot "E3_trocr_base_printed_ipa")
}

if (Test-PythonPackage "torch") {
    $SyllData = Join-Path $Root "ipa_ocr_work\dataset\shaoxing_syllable_ocr"
    $Out = Join-Path $RunRoot "E4_closed_set_syllable_classifier"
    $RowScore = Join-Path $Out "row_score.tsv"
    $Pred = Join-Path $Out "predictions_syllable_crop.tsv"
    $Cmd = @"
$Python "$Root\ipa_ocr_work\scripts\train_syllable_classifier.py" --eval-dir "$SyllData" --out-dir "$Out" --variant syllable_crop --epochs 160 --batch-size $Batch --height 64 --width 160 --lr 0.001 --save-every 20
$Python "$Root\ipa_ocr_work\scripts\score_syllable_ocr_rows.py" --manifest "$SyllData\eval_manifest.tsv" --predictions "$Pred" --out "$RowScore"
"@
    Run-Step "E4_closed_set_syllable_classifier" $Cmd $Out | Out-Null
} else {
    Add-Experiment "E4_closed_set_syllable_classifier" "skipped" "torch unavailable" (Join-Path $RunRoot "E4_closed_set_syllable_classifier")
}

Write-Status "finished"
Write-Host "STATUS=$StatusPath"
