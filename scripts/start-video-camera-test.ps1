[CmdletBinding()]
param(
    [string]$VideoPath,

    [string]$Url = "http://localhost:8000",

    [switch]$ValidateOnly,

    [switch]$PassThru
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $VideoPath) {
    $VideoPath = Join-Path $ProjectRoot "assets\demo\deep-sea-demo-camera.y4m"
}

if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) {
    throw "Y4M video file does not exist: $VideoPath"
}

$resolvedVideo = (Resolve-Path -LiteralPath $VideoPath).Path
if ([System.IO.Path]::GetExtension($resolvedVideo) -ne ".y4m") {
    throw "Chrome fake camera input must be a .y4m file: $resolvedVideo"
}

$stream = [System.IO.File]::OpenRead($resolvedVideo)
try {
    $buffer = New-Object byte[] 128
    $read = $stream.Read($buffer, 0, $buffer.Length)
    $firstLine = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $read).Split("`n")[0]
} finally {
    $stream.Dispose()
}

if (-not $firstLine.StartsWith("YUV4MPEG2 ")) {
    throw "The file does not contain a valid Y4M header: $resolvedVideo"
}

$chromeCandidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
)
$chrome = $chromeCandidates |
    Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
    Select-Object -First 1

if (-not $chrome) {
    throw "Google Chrome was not found."
}

Write-Host "Y4M video: $resolvedVideo"
Write-Host "Y4M header: $firstLine"
Write-Host "Chrome: $chrome"

if ($ValidateOnly) {
    Write-Host "Fake-camera launch validation passed."
    return
}

# A unique profile forces Chrome to create a separate browser process.
# Without it, an already-running normal Chrome instance may consume the URL
# while silently ignoring the fake-camera command-line switches.
$profileRoot = Join-Path ([System.IO.Path]::GetTempPath()) "DeepSeaExplorerChromeProfiles"
$sessionName = "{0}-{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss"), $PID
$profileDir = Join-Path $profileRoot $sessionName
New-Item -ItemType Directory -Path $profileDir -Force | Out-Null

$arguments = @(
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--use-file-for-fake-video-capture=`"$resolvedVideo`"",
    "--user-data-dir=`"$profileDir`"",
    "--no-first-run",
    "--no-default-browser-check",
    "--new-window",
    $Url
)

Write-Host "Starting an isolated Chrome fake-camera session..."
$chromeProcess = Start-Process -FilePath $chrome -ArgumentList $arguments -PassThru

$profileReady = $false
for ($attempt = 0; $attempt -lt 10; $attempt++) {
    if (Test-Path -LiteralPath (Join-Path $profileDir "Local State") -PathType Leaf) {
        $profileReady = $true
        break
    }
    Start-Sleep -Milliseconds 500
}

if (-not $profileReady) {
    throw "Chrome did not initialize the isolated test profile: $profileDir"
}

Write-Host "Fake-camera Chrome started successfully."
Write-Host "Test profile: $profileDir"

if ($PassThru) {
    [PSCustomObject]@{
        ProcessId = $chromeProcess.Id
        ProfileDir = $profileDir
        Mode = "simulated"
    }
}
Write-Host "Open the page and click '开始监测'."
