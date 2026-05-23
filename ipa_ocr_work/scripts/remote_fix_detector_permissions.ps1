$ErrorActionPreference = "Continue"

$ImageDir = "I:\shaoxing_tts_remote\work\shaoxing_dual_model\tone_position_detector\images"
Write-Host "IMAGE_DIR=$ImageDir"
if (-not (Test-Path $ImageDir)) {
    throw "missing image dir"
}

attrib -R "$ImageDir\*" /S
icacls $ImageDir /grant "Administrator:F" /T /C | Out-Null

$Probe = Join-Path $ImageDir "p291_0023_s02.png"
if (Test-Path $Probe) {
    Get-Item -LiteralPath $Probe | Format-List FullName,Length,Attributes
    $stream = [System.IO.File]::OpenRead($Probe)
    Write-Host "open_read_ok length=$($stream.Length)"
    $stream.Close()
}

Write-Host "image_count=$((Get-ChildItem -LiteralPath $ImageDir -File).Count)"
