#
# dominion-live-monitor.ps1
#
# Generic live benchmark monitor for DominionOS.
#
# Features:
#   - Watches bench-serial.log live
#   - Auto-discovers ALL BENCH metrics
#   - No hardcoded benchmark names
#   - Shows newest metrics instantly
#   - Works with future benchmark additions
#   - Never exits until Ctrl+C
#

param(
    [string]$LogFile = "..\..\bench-serial.log",
    [int]$RefreshMs = 250
)

$ErrorActionPreference = "SilentlyContinue"

$LogFile = Resolve-Path $LogFile

$script:Metrics = [ordered]@{}
$script:BenchCount = 0
$script:LastUpdate = $null

function Parse-BenchLine {
    param([string]$Line)

    if ($Line -notmatch '^BENCH\s+(\S+)\s+(.*)$') {
        return
    }

    $benchName = $matches[1]
    $rest = $matches[2]

    foreach ($token in ($rest -split '\s+')) {

        if ($token -match '^([^=]+)=([^=]+)$') {

            $metric = $matches[1]
            $value = $matches[2]

            $key = "$benchName.$metric"

            $script:Metrics[$key] = $value
        }
    }

    $script:BenchCount++
    $script:LastUpdate = Get-Date
}

function DrawScreen {

    Clear-Host

    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host "              DominionOS Live Benchmark Monitor"
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host ("Time          : {0}" -f (Get-Date))
    Write-Host ("Log           : {0}" -f $LogFile)
    Write-Host ("Bench Lines   : {0}" -f $script:BenchCount)

    if ($script:LastUpdate) {
        Write-Host ("Last Update   : {0}" -f $script:LastUpdate)
    }

    Write-Host ("Metrics Found : {0}" -f $script:Metrics.Count)

    Write-Host ""
    Write-Host ("{0,-55} {1}" -f "Metric", "Value")
    Write-Host ("-" * 90)

    foreach ($item in $script:Metrics.GetEnumerator() | Sort-Object Name) {

        Write-Host ("{0,-55} {1}" -f $item.Key, $item.Value)
    }

    Write-Host ""
    Write-Host "Monitoring live output..."
    Write-Host "Press Ctrl+C to quit."
}

#
# Initial parse
#
if (Test-Path $LogFile) {

    Get-Content $LogFile | ForEach-Object {
        Parse-BenchLine $_
    }
}

DrawScreen

#
# Track current file position
#
$script:LastPosition = 0

if (Test-Path $LogFile) {
    $script:LastPosition = (Get-Item $LogFile).Length
}

#
# Watch for changes
#
$watcher = New-Object IO.FileSystemWatcher

$watcher.Path = Split-Path $LogFile
$watcher.Filter = Split-Path $LogFile -Leaf
$watcher.NotifyFilter = [IO.NotifyFilters]'Size,LastWrite'
$watcher.EnableRaisingEvents = $true

Register-ObjectEvent `
    -InputObject $watcher `
    -EventName Changed `
    -SourceIdentifier DominionBench `
    -Action {

        try {

            Start-Sleep -Milliseconds 20

            $fs = [System.IO.File]::Open(
                $using:LogFile,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read,
                [System.IO.FileShare]::ReadWrite
            )

            try {

                $fs.Seek($script:LastPosition, [System.IO.SeekOrigin]::Begin) | Out-Null

                $reader = New-Object IO.StreamReader($fs)

                while (-not $reader.EndOfStream) {

                    $line = $reader.ReadLine()

                    if ($line) {
                        Parse-BenchLine $line
                    }
                }

                $script:LastPosition = $fs.Position
            }
            finally {

                $reader.Dispose()
                $fs.Dispose()
            }

            DrawScreen
        }
        catch {
        }
    } | Out-Null

Write-Host ""
Write-Host "Live monitoring started." -ForegroundColor Green
Write-Host ""

while ($true) {
    Start-Sleep -Milliseconds $RefreshMs
}