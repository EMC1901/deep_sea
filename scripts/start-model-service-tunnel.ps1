[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9.-]+$')]
    [string]$ServerHost,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z_][A-Za-z0-9_-]*$')]
    [string]$ServerUser,

    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,

    [ValidateRange(1, 65535)]
    [int]$LocalPort = 19000,

    [ValidateRange(1, 65535)]
    [int]$RemotePort = 19000
)

$ErrorActionPreference = "Stop"

$occupied = Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction SilentlyContinue
if ($occupied) {
    throw "Local port $LocalPort is already listening. Choose another -LocalPort or stop the existing listener."
}

Write-Host "Opening a local-only tunnel: 127.0.0.1:$LocalPort -> server 127.0.0.1:$RemotePort"
Write-Host "The SSH client may prompt for host-key verification and authentication. Keep this window open."

& ssh `
    -N `
    -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=3 `
    -L "${LocalPort}:127.0.0.1:${RemotePort}" `
    -p $SshPort `
    "${ServerUser}@${ServerHost}"

if ($LASTEXITCODE -ne 0) {
    throw "The SSH tunnel ended with exit code $LASTEXITCODE."
}
