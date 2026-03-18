param(
    [int]$ApiPort = 8100,
    [int]$AppPort = 8602
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$apiScript = (Resolve-Path (Join-Path $PSScriptRoot "serve_api.ps1")).Path
$streamlitExe = (Resolve-Path (Join-Path $repoRoot "Backend\.venv311\Scripts\streamlit.exe")).Path

function Start-DetachedPowerShell {
    param(
        [string]$ScriptPath,
        [int]$Port
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = '-NoExit -NoProfile -ExecutionPolicy Bypass -File "' + $ScriptPath + '" -Port ' + $Port
    $psi.UseShellExecute = $true
    [void][System.Diagnostics.Process]::Start($psi)
}

Start-DetachedPowerShell -ScriptPath $apiScript -Port $ApiPort

Start-Sleep -Seconds 3

Write-Host "API URL: http://127.0.0.1:$ApiPort"
Write-Host "Presentation URL: http://127.0.0.1:$AppPort"
Write-Host "The API runs in a separate PowerShell window."
Write-Host "The presentation app will run in this window. Keep it open while using the app."

Set-Location $repoRoot
& $streamlitExe run Backend\sea_level_risk\presentation_app.py --server.port $AppPort --server.headless true
