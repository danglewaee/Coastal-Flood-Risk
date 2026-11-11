param([int]$Port = 8100)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $repoRoot
try {
    & (Join-Path $repoRoot 'Backend\.venv311\Scripts\python.exe') -m Backend.sea_level_risk.explorer_api --port $Port
} finally {
    Pop-Location
}
