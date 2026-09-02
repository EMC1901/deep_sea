[CmdletBinding()]
param(
    [string]$SessionId,
    [string]$SshHost = "deepsea-codex",
    [string]$RemoteProjectRoot = "/projects/deep-sea-explorer-codex/app",
    [string]$OutputRoot = "C:\Users\emc20\Downloads"
)

$ErrorActionPreference = "Stop"

if ($SshHost -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "SshHost has an invalid format."
}
if ($RemoteProjectRoot -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "RemoteProjectRoot has an invalid format."
}
if ($SessionId -and $SessionId -notmatch '^[A-Za-z0-9_-]+$') {
    throw "SessionId has an invalid format."
}

$ssh = Get-Command ssh.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
$scp = Get-Command scp.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ssh -or $null -eq $scp) {
    throw "Windows OpenSSH ssh.exe and scp.exe are required."
}
if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
    throw "The local output directory does not exist: $OutputRoot"
}

$remoteCommand = "cd -- $RemoteProjectRoot && ./scripts/server/export-keyframe-retrieval-review.sh"
if ($SessionId) {
    $remoteCommand += " --session-id $SessionId"
}
$output = & $ssh.Source $SshHost $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "The server-side retrieval review export failed."
}
$pathLine = $output | Where-Object { $_ -like "REVIEW_EXPORT_PATH=*" } | Select-Object -Last 1
if (-not $pathLine) {
    throw "The server did not return a review export path."
}
$remotePath = $pathLine.Substring("REVIEW_EXPORT_PATH=".Length)
if ($remotePath -notmatch '^/projects/deep-sea-explorer-codex/app/runtime/retrieval-review-exports/review\.[A-Za-z0-9]+$') {
    throw "The server returned an invalid review export path."
}

$localName = "DeepSeaExplorer-RetrievalReview-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$localPath = Join-Path $OutputRoot $localName
& $scp.Source -r "${SshHost}:$remotePath" $localPath
if ($LASTEXITCODE -ne 0) {
    throw "Downloading the review export failed. The server-side export was retained for retry."
}

Write-Host "Review export downloaded to: $localPath"
if ($env:DEEP_SEA_AUTOMATION_TEST -ne "1") {
    Invoke-Item -LiteralPath $localPath
}
