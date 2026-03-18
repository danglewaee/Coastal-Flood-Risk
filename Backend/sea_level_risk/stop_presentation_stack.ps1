param(
    [int]$ApiPort = 8100,
    [int]$AppPort = 8602
)

$ErrorActionPreference = "Stop"

$targets = @($ApiPort, $AppPort)
$listeners = netstat -ano -p tcp | Select-String -Pattern ($targets | ForEach-Object { ":{0}\s" -f $_ })
$pids = @()

foreach ($line in $listeners) {
    $parts = ($line.ToString() -split "\s+") | Where-Object { $_ }
    if ($parts.Count -ge 5) {
        $pids += [int]$parts[-1]
    }
}

$pids = $pids | Sort-Object -Unique

if (-not $pids) {
    Write-Host "No listeners found on ports $ApiPort or $AppPort."
    exit 0
}

foreach ($targetPid in $pids) {
    try {
        Stop-Process -Id $targetPid -Force -ErrorAction Stop
        Write-Host "Stopped process $targetPid"
    } catch {
        Write-Warning "Failed to stop process ${targetPid}: $($_.Exception.Message)"
    }
}
