param(
    [int]$Port = 8602
)

$ErrorActionPreference = "Stop"

try {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
    $streamlitExe = (Resolve-Path (Join-Path $repoRoot "Backend\.venv311\Scripts\streamlit.exe")).Path
    Set-Location $repoRoot
    & $streamlitExe run Backend\sea_level_risk\presentation_app.py --server.port $Port --server.headless true
} catch {
    Write-Error $_
    Read-Host "Presentation app failed. Press Enter to close"
}
