[CmdletBinding()]
param(
    [string]$ServerAddress,

    [string]$SshUser,

    [int]$SshPort = 0
)

$ErrorActionPreference = "Stop"
$LocalPort = 19100
$RemotePort = 19100
$LoopbackAddress = "127.0.0.1"
$LocalForward = "127.0.0.1:19100:127.0.0.1:19100"

function Show-Usage {
    Write-Host "Usage:"
    Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" ``"
    Write-Host "    -ServerAddress <server-host-or-ip> ``"
    Write-Host "    -SshUser <ssh-user> ``"
    Write-Host "    -SshPort <1-65535>"
    Write-Host ""
    Write-Host "This opens only:"
    Write-Host "  Windows 127.0.0.1:19100 -> SSH -> server 127.0.0.1:19100"
    Write-Host ""
    Write-Host "The server address, SSH user, credentials, and private-key path are not stored by this script."
}

function Test-RequiredParameters {
    $missing = @()
    if ([string]::IsNullOrWhiteSpace($ServerAddress)) {
        $missing += "-ServerAddress"
    }
    if ([string]::IsNullOrWhiteSpace($SshUser)) {
        $missing += "-SshUser"
    }
    if ($SshPort -eq 0) {
        $missing += "-SshPort"
    }

    if ($missing.Count -gt 0) {
        Write-Host "Missing required parameters: $($missing -join ', ')" -ForegroundColor Yellow
        Show-Usage
        return $false
    }

    if ($SshPort -lt 1 -or $SshPort -gt 65535) {
        Write-Host "Invalid -SshPort: expected a value from 1 to 65535." -ForegroundColor Red
        Show-Usage
        return $false
    }

    if ($SshUser -notmatch '^[A-Za-z0-9_][A-Za-z0-9._-]*$') {
        Write-Host "Invalid -SshUser format." -ForegroundColor Red
        return $false
    }

    if ($ServerAddress -notmatch '^[A-Za-z0-9][A-Za-z0-9._:-]*$') {
        Write-Host "Invalid -ServerAddress format." -ForegroundColor Red
        return $false
    }

    return $true
}

function Test-LoopbackPortAvailable {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            $LocalPort
        )
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
        return $true
    }
    catch [System.Net.Sockets.SocketException] {
        return $false
    }
    finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

if (-not (Test-RequiredParameters)) {
    exit 2
}

if (-not (Test-LoopbackPortAvailable)) {
    Write-Host "Local port $LoopbackAddress`:$LocalPort is already in use. Tunnel not started." -ForegroundColor Red
    exit 3
}

$sshCommand = Get-Command ssh.exe -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -eq $sshCommand) {
    $sshCommand = Get-Command ssh -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
}
if ($null -eq $sshCommand) {
    Write-Host "OpenSSH client was not found. Install or enable the Windows OpenSSH Client first." -ForegroundColor Red
    exit 4
}

$destination = "$SshUser@$ServerAddress"
$sshArguments = @(
    "-N",
    "-T",
    "-p", $SshPort.ToString(),
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", $LocalForward,
    $destination
)

Write-Host "Starting an SSH API tunnel in the foreground."
Write-Host "Local endpoint:  $LoopbackAddress`:$LocalPort"
Write-Host "Remote endpoint: $LoopbackAddress`:$RemotePort"
Write-Host "Press Ctrl+C in this window to close the tunnel."

& $sshCommand.Source @sshArguments
$sshExitCode = $LASTEXITCODE

if ($sshExitCode -ne 0) {
    Write-Host "SSH tunnel exited with code $sshExitCode." -ForegroundColor Red
    exit $sshExitCode
}

exit 0
