[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("test", "lint", "api", "speech", "web", "remote-model-check")]
    [string]$Action,

    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src" + $(if ($env:PYTHONPATH) { ";$($env:PYTHONPATH)" } else { "" })

function Get-PythonCommand {
    if ($env:PYTHON) {
        return $env:PYTHON
    }
    return "python"
}

function Import-LocalEnvFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }

        $parts = $trimmed.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim([char]34).Trim([char]39)
        if ($name -match '^[A-Za-z_][A-Za-z0-9_]*$') {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

function Assert-RemoteDevelopmentMode {
    if ($env:MODEL_BACKEND -ne "remote") {
        throw "Development commands require MODEL_BACKEND=remote; current value: '$($env:MODEL_BACKEND)'."
    }
}

Import-LocalEnvFile -Path $EnvFile
$Python = Get-PythonCommand

switch ($Action) {
    "test" {
        # Default tests use fake mode and must not access model services or the network.
        $env:MODEL_BACKEND = "fake"
        & $Python -B -m pytest -q
    }
    "lint" {
        & $Python -m ruff check .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python -m ruff format --check .
    }
    "api" {
        Assert-RemoteDevelopmentMode
        & $Python -B -m deep_sea_explorer.main
    }
    "speech" {
        & $Python -B -m deep_sea_explorer.speech_main
    }
    "web" {
        & $Python -B -m http.server 8000 --directory frontend
    }
    "remote-model-check" {
        Assert-RemoteDevelopmentMode
        if ($env:MODEL_SERVICE_ENABLED -ne "true") {
            throw "Model service checks are disabled. Set MODEL_SERVICE_ENABLED=true in a local .env before explicitly running this action."
        }
        if (-not $env:MODEL_SERVICE_BASE_URL -or $env:MODEL_SERVICE_BASE_URL -match '\.invalid/?$') {
            throw "MODEL_SERVICE_BASE_URL must be a real server address, not a placeholder."
        }
        if ($env:MODEL_SERVICE_AUTH_TYPE -eq "bearer" -and -not $env:MODEL_SERVICE_AUTH_TOKEN) {
            throw "Bearer authentication requires MODEL_SERVICE_AUTH_TOKEN. This script never prints the token."
        }

        $baseUrl = $env:MODEL_SERVICE_BASE_URL.TrimEnd('/')
        $apiPrefix = $env:MODEL_SERVICE_API_PREFIX
        if (-not $apiPrefix) {
            $apiPrefix = "/v1"
        }
        $prefix = $apiPrefix.Trim('/')
        $headers = @{}
        if ($env:MODEL_SERVICE_AUTH_TYPE -eq "bearer") {
            $headers["Authorization"] = "Bearer $($env:MODEL_SERVICE_AUTH_TOKEN)"
        }
        $connectTimeout = $env:MODEL_SERVICE_CONNECT_TIMEOUT_SECONDS
        if (-not $connectTimeout) {
            $connectTimeout = 5
        }
        $timeout = [Math]::Max(1, [int]$connectTimeout)
        $healthUrl = "$baseUrl/$prefix/health"
        $response = Invoke-RestMethod -Method Get -Uri $healthUrl -Headers $headers -TimeoutSec $timeout
        Write-Host "Remote model health check succeeded: $healthUrl"
        $response | ConvertTo-Json -Depth 8
    }
}
