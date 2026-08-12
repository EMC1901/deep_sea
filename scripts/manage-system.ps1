[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action,

    [ValidateSet("real", "simulated")]
    [string]$Mode = "real",

    [string]$VideoPath,
    [string]$SshHost = "deepsea-codex",
    [string]$RemoteProjectRoot = "/projects/deep-sea-explorer-codex/app",
    [int]$WebPort = 19100,
    [int]$ApiPort = 9001,
    [int]$SpeechPort = 9009
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot "runtime"
$TunnelPidFile = Join-Path $RuntimeDir "system-tunnel.pid"
$BrowserStateFile = Join-Path $RuntimeDir "system-browser.json"

function Assert-LocalPort {
    param([int]$Port)

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
    } catch [System.Net.Sockets.SocketException] {
        throw "Local port 127.0.0.1:$Port is already in use. Refusing to replace its owner."
    } finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

function Get-ManagedProcess {
    param([int]$ProcessId, [string[]]$ExpectedArguments)

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $null
    }
    $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId").CommandLine
    if ([string]::IsNullOrWhiteSpace($commandLine)) {
        return $null
    }
    foreach ($expected in $ExpectedArguments) {
        if ($commandLine.IndexOf($expected, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
            return $null
        }
    }
    return $process
}

function Read-ManagedPid {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    $value = (Get-Content -LiteralPath $Path -Raw).Trim()
    if ($value -notmatch '^[1-9][0-9]*$') {
        return $null
    }
    return [int]$value
}

function Stop-ManagedProcess {
    param([string]$Path, [string[]]$ExpectedArguments, [string]$Name)

    $processId = Read-ManagedPid -Path $Path
    if ($null -eq $processId) {
        return
    }
    $process = Get-ManagedProcess -ProcessId $processId -ExpectedArguments $ExpectedArguments
    if ($null -eq $process) {
        Write-Warning "Refusing to stop ${Name}: its saved PID is stale or does not match the managed command."
        return
    }
    Stop-Process -Id $processId -ErrorAction Stop
    Remove-Item -LiteralPath $Path -Force
    Write-Host "Stopped $Name (PID $processId)."
}

function Get-Chrome {
    $candidates = @(
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
    )
    $chrome = $candidates |
        Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        Select-Object -First 1
    if (-not $chrome) {
        throw "Google Chrome was not found."
    }
    return $chrome
}

function Invoke-RemoteSystemAction {
    param([string]$RemoteAction)

    if ($SshHost -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
        throw "SshHost has an invalid format."
    }
    if ($RemoteProjectRoot -notmatch '^/[A-Za-z0-9._/-]+$') {
        throw "RemoteProjectRoot has an invalid format."
    }
    $ssh = Get-Command ssh.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $ssh) {
        throw "Windows OpenSSH client (ssh.exe) was not found."
    }
    $remoteCommand = "cd -- $RemoteProjectRoot && ./scripts/server/manage-development-system.sh $RemoteAction"
    & $ssh.Source $SshHost $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote system action '$RemoteAction' failed."
    }
}

function Get-TunnelExpectedArguments {
    return @(
        "-L",
        "127.0.0.1:$WebPort`:127.0.0.1:19100",
        "127.0.0.1:$ApiPort`:127.0.0.1:9001",
        "127.0.0.1:$SpeechPort`:127.0.0.1:9009",
        $SshHost
    )
}

function Find-ExistingTunnel {
    $expected = Get-TunnelExpectedArguments
    $savedPid = Read-ManagedPid -Path $TunnelPidFile
    if ($null -ne $savedPid) {
        $savedProcess = Get-ManagedProcess -ProcessId $savedPid -ExpectedArguments $expected
        if ($null -ne $savedProcess) {
            return $savedProcess
        }
    }
    $matches = @(
        Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'" |
            ForEach-Object {
                Get-ManagedProcess -ProcessId $_.ProcessId -ExpectedArguments $expected
            } |
            Where-Object { $null -ne $_ }
    )
    if ($matches.Count -eq 1) {
        return $matches[0]
    }
    return $null
}

function Start-Tunnel {
    $existing = Find-ExistingTunnel
    if ($null -ne $existing) {
        Set-Content -LiteralPath $TunnelPidFile -Value $existing.Id -NoNewline
        Write-Host "SSH tunnel is already running (PID $($existing.Id))."
        return
    }
    foreach ($port in @($WebPort, $ApiPort, $SpeechPort)) {
        Assert-LocalPort -Port $port
    }

    $ssh = Get-Command ssh.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $ssh) {
        throw "Windows OpenSSH client (ssh.exe) was not found."
    }
    $arguments = @(
        "-N", "-T",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-L", "127.0.0.1:$WebPort`:127.0.0.1:19100",
        "-L", "127.0.0.1:$ApiPort`:127.0.0.1:9001",
        "-L", "127.0.0.1:$SpeechPort`:127.0.0.1:9009",
        $SshHost
    )
    $process = Start-Process -FilePath $ssh.Source -ArgumentList $arguments -PassThru -WindowStyle Hidden
    Set-Content -LiteralPath $TunnelPidFile -Value $process.Id -NoNewline
    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        Remove-Item -LiteralPath $TunnelPidFile -Force
        throw "SSH tunnel exited before it became ready."
    }
    Write-Host "Started SSH tunnel (PID $($process.Id))."
}

function Wait-ForApi {
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
            if ($response.status -eq "ok") {
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "API health endpoint did not become ready through the SSH tunnel."
}

function Start-RealBrowser {
    $chrome = Get-Chrome
    $profileRoot = Join-Path ([System.IO.Path]::GetTempPath()) "DeepSeaExplorerSystemProfiles"
    $profileDir = Join-Path $profileRoot ("real-{0}-{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss"), $PID)
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    $process = Start-Process -FilePath $chrome -ArgumentList @(
        "--user-data-dir=`"$profileDir`"",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
        "http://127.0.0.1:$WebPort/"
    ) -PassThru
    return [PSCustomObject]@{ ProcessId = $process.Id; ProfileDir = $profileDir; Mode = "real" }
}

function Start-SimulatedBrowser {
    if (-not $VideoPath) {
        $VideoPath = Join-Path $ProjectRoot "assets\demo\deep-sea-demo-camera.y4m"
    }
    $launcher = Join-Path $PSScriptRoot "start-video-camera-test.ps1"
    $state = & $launcher -VideoPath $VideoPath -Url "http://127.0.0.1:$WebPort/" -PassThru
    if ($null -eq $state -or $null -eq $state.ProcessId) {
        throw "The simulated-camera browser did not provide a managed process ID."
    }
    return $state
}

function Stop-ManagedBrowser {
    if (-not (Test-Path -LiteralPath $BrowserStateFile -PathType Leaf)) {
        return
    }
    $browser = Get-Content -LiteralPath $BrowserStateFile -Raw | ConvertFrom-Json
    $browserPidFile = Join-Path $RuntimeDir "system-browser.pid"
    Set-Content -LiteralPath $browserPidFile -Value $browser.ProcessId -NoNewline
    Stop-ManagedProcess -Path $browserPidFile -ExpectedArguments @($browser.ProfileDir) -Name "browser"
    Remove-Item -LiteralPath $browserPidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $BrowserStateFile -Force -ErrorAction SilentlyContinue
}

switch ($Action) {
    "start" {
        New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
        $existingTunnel = Find-ExistingTunnel
        if ($null -eq $existingTunnel) {
            foreach ($port in @($WebPort, $ApiPort, $SpeechPort)) {
                Assert-LocalPort -Port $port
            }
        }
        Invoke-RemoteSystemAction -RemoteAction "start"
        Start-Tunnel
        try {
            Wait-ForApi
            Stop-ManagedBrowser
            $browser = if ($Mode -eq "simulated") { Start-SimulatedBrowser } else { Start-RealBrowser }
            $browser | ConvertTo-Json | Set-Content -LiteralPath $BrowserStateFile -NoNewline
            Write-Host "System started in $Mode mode: http://127.0.0.1:$WebPort/"
        } catch {
            Stop-ManagedProcess -Path $TunnelPidFile -ExpectedArguments (Get-TunnelExpectedArguments) -Name "SSH tunnel"
            Invoke-RemoteSystemAction -RemoteAction "stop"
            throw
        }
    }
    "stop" {
        Invoke-RemoteSystemAction -RemoteAction "stop"
        Stop-ManagedBrowser
        $existingTunnel = Find-ExistingTunnel
        if ($null -ne $existingTunnel) {
            Set-Content -LiteralPath $TunnelPidFile -Value $existingTunnel.Id -NoNewline
        }
        Stop-ManagedProcess -Path $TunnelPidFile -ExpectedArguments (Get-TunnelExpectedArguments) -Name "SSH tunnel"
    }
    "status" {
        Invoke-RemoteSystemAction -RemoteAction "status"
        $tunnel = Find-ExistingTunnel
        if ($null -ne $tunnel) {
            Write-Host "SSH tunnel: running (PID $($tunnel.Id))"
        } else {
            Write-Host "SSH tunnel: stopped or unmanaged"
        }
    }
}
