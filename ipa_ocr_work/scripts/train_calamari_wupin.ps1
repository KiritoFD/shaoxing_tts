param(
    [string]$Dataset = "ipa_ocr_work\dataset\shaoxing_wupin",
    [string]$OutputDir = "ipa_ocr_work\models\calamari_wupin",
    [int]$Epochs = 50
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command calamari-train -ErrorAction SilentlyContinue)) {
    Write-Host "calamari-train was not found in PATH."
    Write-Host "Create/activate a Calamari environment first, for example:"
    Write-Host "  conda create -n calamari_wupin python=3.10 -y"
    Write-Host "  conda activate calamari_wupin"
    Write-Host "  pip install calamari-ocr tensorflow==2.15.0 protobuf==3.20.3"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

calamari-train `
    --trainer.output_dir $OutputDir `
    --train.images "$Dataset\train\images\*.png" `
    --val.images "$Dataset\val\images\*.png" `
    --trainer.epochs $Epochs `
    --early_stopping.frequency 1 `
    --early_stopping.n_to_go 8
