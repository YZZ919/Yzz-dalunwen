param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][string]$Output,
    [int64]$TotalBytes = 0,
    [int64]$ChunkBytes = 67108864,
    [int]$Workers = 8
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $Output
$name = [IO.Path]::GetFileName($Output)
$chunkDir = Join-Path $root ("." + $name + ".chunks")
New-Item -ItemType Directory -Force -Path $chunkDir | Out-Null

if ($TotalBytes -le 0) {
    $head = & curl.exe -sS -L -I --max-time 120 $Url
    $line = $head | Select-String -Pattern 'Content-Range:.*?/([0-9]+)' | Select-Object -First 1
    if (-not $line) { throw "Could not determine remote size" }
    $TotalBytes = [int64]$line.Matches[0].Groups[1].Value
}

$jobs = @()
for ($start = 0; $start -lt $TotalBytes; $start += $ChunkBytes) {
    $end = [Math]::Min($TotalBytes - 1, $start + $ChunkBytes - 1)
    $idx = [int]($start / $ChunkBytes)
    $part = Join-Path $chunkDir ("part_{0:D5}.bin" -f $idx)
    $expected = $end - $start + 1
    if ((Test-Path -LiteralPath $part) -and ((Get-Item -LiteralPath $part).Length -eq $expected)) { continue }
    while (@($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $Workers) {
        $done = Wait-Job -Job $jobs -Any -Timeout 5
        if ($done) { Receive-Job -Job $done -Keep | Write-Output; Remove-Job -Job $done }
        $jobs = @($jobs | Where-Object { $_.State -eq 'Running' })
    }
    $jobs += Start-Job -ScriptBlock {
        param($Url,$part,$start,$end,$expected)
        $ErrorActionPreference = 'Stop'
        & curl.exe -sS -L --fail --retry 8 --retry-delay 2 -H ("Range: bytes={0}-{1}" -f $start,$end) -o $part $Url
        if (-not (Test-Path -LiteralPath $part) -or ((Get-Item -LiteralPath $part).Length -ne $expected)) {
            throw "Bad chunk $start-$end"
        }
        "DONE $start-$end"
    } -ArgumentList $Url,$part,$start,$end,$expected
}
while ($jobs.Count -gt 0) {
    $done = Wait-Job -Job $jobs -Any -Timeout 5
    if ($done) { Receive-Job -Job $done; if ($done.State -eq 'Failed') { throw $done.ChildJobs[0].JobStateInfo.Reason }; Remove-Job -Job $done }
    $jobs = @($jobs | Where-Object { $_.State -eq 'Running' })
}

$tmp = "$Output.partial"
if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
$stream = [IO.File]::Open($tmp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    for ($start = 0; $start -lt $TotalBytes; $start += $ChunkBytes) {
        $idx = [int]($start / $ChunkBytes)
        $part = Join-Path $chunkDir ("part_{0:D5}.bin" -f $idx)
        $bytes = [IO.File]::ReadAllBytes($part)
        $stream.Write($bytes, 0, $bytes.Length)
    }
} finally { $stream.Dispose() }
Move-Item -LiteralPath $tmp -Destination $Output -Force
Write-Output "Downloaded $TotalBytes bytes to $Output"
