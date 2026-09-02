[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ServerHost,

    [Parameter(Mandatory = $true)]
    [string]$SshUser,

    [Parameter(Mandatory = $true)]
    [string]$IdentityFile,

    [int]$SshPort = 22,
    [int]$LocalPort = 19100,
    [int]$RemotePort = 19100
)

$ErrorActionPreference = "Stop"

if ($ServerHost -notmatch '^[A-Za-z0-9][A-Za-z0-9._:-]*$') {
    throw "ServerHost has an invalid format."
}
if ($SshUser -notmatch '^[A-Za-z0-9_][A-Za-z0-9._-]*$') {
    throw "SshUser has an invalid format."
}
if ($SshPort -lt 1 -or $SshPort -gt 65535 -or $LocalPort -lt 1 -or $LocalPort -gt 65535 -or $RemotePort -lt 1 -or $RemotePort -gt 65535) {
    throw "SSH and tunnel ports must be between 1 and 65535."
}
if (-not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) {
    throw "IdentityFile does not exist: $IdentityFile"
}

$listener = $null
try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $LocalPort)
    $listener.Server.ExclusiveAddressUse = $true
    $listener.Start()
} catch [System.Net.Sockets.SocketException] {
    throw "Local port 127.0.0.1:$LocalPort is already in use."
} finally {
    if ($null -ne $listener) {
        $listener.Stop()
    }
}

$ssh = Get-Command ssh.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ssh) {
    throw "Windows OpenSSH client (ssh.exe) was not found."
}

$destination = "$SshUser@$ServerHost"
$forward = "127.0.0.1:$LocalPort`:127.0.0.1:$RemotePort"
$arguments = @(
    "-N",
    "-T",
    "-i", $IdentityFile,
    "-p", $SshPort.ToString(),
    "-o", "BatchMode=yes",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", $forward,
    $destination
)

Write-Host "Private tunnel ready after SSH connects: http://127.0.0.1:$LocalPort/"
Write-Host "Press Ctrl+C in this window to close the tunnel."
& $ssh.Source @arguments
exit $LASTEXITCODE
