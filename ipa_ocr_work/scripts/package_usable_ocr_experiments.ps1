param(
    [string]$Root = "",
    [string]$Out = "",
    [switch]$UseCompressArchive
)

$ErrorActionPreference = "Stop"

if (-not $Root) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
if (-not $Out) {
    $Out = Join-Path $Root "ipa_ocr_work\usable_ocr_experiment_payload.zip"
}

$items = @(
    "README.md",
    "result_all_converted.clean.csv",
    "result_all_converted.with_ipa.csv",
    "docs\ocr_usable_path.md",
    "docs\shaoxing_wupin_to_ipa.md",
    "docs\structured_tone_label_schema.md",
    "docs\tone_position_detector_plan.md",
    "ipa_ocr_work\config",
    "ipa_ocr_work\scripts",
    "ipa_ocr_work\dataset\shaoxing_pdf136_clean",
    "ipa_ocr_work\dataset\shaoxing_ipa_matched_skip3",
    "ipa_ocr_work\dataset\shaoxing_dual_model",
    "ipa_ocr_work\dataset\shaoxing_syllable_ocr",
    "ipa_ocr_work\dataset\shaoxing_tone_detector_clustered_v2"
)

if (Test-Path $Out) {
    Remove-Item -LiteralPath $Out -Force
}
$existing = @()
foreach ($item in $items) {
    if (Test-Path (Join-Path $Root $item)) {
        $existing += $item
    } else {
        Write-Warning "skip missing $item"
    }
}
if (-not $existing) {
    throw "No payload items found"
}

if (-not $UseCompressArchive -and (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
    Push-Location $Root
    try {
        if ($Out.EndsWith(".zip")) {
            & tar.exe -a -cf $Out @existing
        } else {
            & tar.exe -czf $Out @existing
        }
        if ($LASTEXITCODE -ne 0) {
            throw "tar failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} else {
    $temp = Join-Path $env:TEMP ("shaoxing_ocr_payload_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $temp | Out-Null
    foreach ($item in $existing) {
        $src = Join-Path $Root $item
        $dst = Join-Path $temp $item
        $parent = Split-Path $dst -Parent
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        if ((Get-Item $src).PSIsContainer) {
            Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
        } else {
            Copy-Item -LiteralPath $src -Destination $dst -Force
        }
    }
    Compress-Archive -Path (Join-Path $temp "*") -DestinationPath $Out -Force
    Remove-Item -LiteralPath $temp -Recurse -Force
}
Write-Host "wrote $Out"
