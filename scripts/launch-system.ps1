[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "stop")]
    [string]$Action,

    [ValidateSet("real", "simulated")]
    [string]$Mode = "real",

    [string]$VideoPath
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Manager = Join-Path $PSScriptRoot "manage-system.ps1"

function Show-Message {
    param([string]$Text, [System.Windows.Forms.MessageBoxIcon]$Icon)

    if ($env:DEEP_SEA_AUTOMATION_TEST -eq "1") {
        Write-Output $Text
        return
    }
    [void][System.Windows.Forms.MessageBox]::Show(
        $Text,
        "Deep Sea Explorer",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        $Icon
    )
}

function Select-SimulationVideo {
    $dialog = [System.Windows.Forms.OpenFileDialog]::new()
    $dialog.Title = "Select a Y4M video for the simulated camera"
    $dialog.Filter = "Y4M video (*.y4m)|*.y4m|All files (*.*)|*.*"
    $runtimeDirectory = Join-Path $ProjectRoot "runtime"
    $dialog.InitialDirectory = if (Test-Path -LiteralPath $runtimeDirectory) {
        $runtimeDirectory
    } else {
        $ProjectRoot
    }
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        return $null
    }
    return $dialog.FileName
}

try {
    if ($Action -eq "stop") {
        & $Manager stop
        Show-Message -Text "The system is closed. Monitoring data, reports, and model files are preserved." -Icon Information
        return
    }

    if ($Mode -eq "simulated") {
        $videoPath = $VideoPath
        if (-not $videoPath) {
            $videoPath = Select-SimulationVideo
        }
        if (-not $videoPath) {
            return
        }
        & $Manager start -Mode $Mode -VideoPath $videoPath
    } else {
        & $Manager start -Mode $Mode
    }
    $modeText = if ($Mode -eq "simulated") { "simulated camera" } else { "real camera" }
    Show-Message -Text "The system started with the $modeText. The browser is open; click Start Monitoring to begin." -Icon Information
} catch {
    Show-Message -Text "The system operation did not complete: $($_.Exception.Message)" -Icon Error
    exit 1
}
