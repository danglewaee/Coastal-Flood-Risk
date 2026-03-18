param(
    [int]$Port = 8100
)

$ErrorActionPreference = "Stop"

try {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
    $pythonExe = (Resolve-Path (Join-Path $repoRoot "Backend\.venv311\Scripts\python.exe")).Path
    Set-Location $repoRoot
    & $pythonExe -m Backend.sea_level_risk.realtime_api --host 127.0.0.1 --port $Port
} catch {
    Write-Error $_
    Read-Host "API process failed. Press Enter to close"
}
