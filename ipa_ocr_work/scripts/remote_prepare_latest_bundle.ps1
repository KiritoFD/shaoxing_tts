$ErrorActionPreference = "Stop"

$ArchiveCandidates = @(
    "C:\Users\Administrator\shaoxing_tts_remote\remote_training_bundle_latest.tar.gz",
    "C:\Users\Administrator\remote_training_bundle_latest.tar.gz"
)
$Archive = $ArchiveCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Archive) {
    throw "remote_training_bundle_latest.tar.gz not found"
}

$Root = "I:\shaoxing_tts_remote\work_latest"
if (-not (Test-Path "I:\")) {
    $Root = "C:\Users\Administrator\shaoxing_tts_remote\work_latest"
}

if (Test-Path $Root) {
    Remove-Item -LiteralPath $Root -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Root | Out-Null
tar -xzf $Archive -C $Root

Write-Host "ROOT=$Root"
Get-ChildItem -LiteralPath $Root -Force | Select-Object Name,Length,LastWriteTime
