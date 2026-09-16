param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][string]$DatasetRoot
)

$ErrorActionPreference = 'Stop'
$archivePath = [IO.Path]::GetFullPath($Archive)
$datasetPath = [IO.Path]::GetFullPath($DatasetRoot)
$bundlePath = Join-Path $datasetPath '_bundles'
New-Item -ItemType Directory -Path $bundlePath -Force | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem
$outer = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $innerEntries = @($outer.Entries | Where-Object { $_.FullName.EndsWith('.zip') })
    if ($innerEntries.Count -eq 0) {
        throw "No inner ASL zip files found in $archivePath"
    }
    foreach ($entry in $innerEntries) {
        $name = [IO.Path]::GetFileName($entry.FullName)
        $target = Join-Path $bundlePath $name
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        Write-Output "Extracted inner archive: $name"
    }
} finally {
    $outer.Dispose()
}

Get-ChildItem -LiteralPath $bundlePath -Filter '*.zip' | Sort-Object Name | ForEach-Object {
    $sequence = [IO.Path]::GetFileNameWithoutExtension($_.Name)
    $target = Join-Path $datasetPath $sequence
    $ready = Join-Path $target 'mav0\cam0\data.csv'
    if (Test-Path -LiteralPath $ready) {
        Write-Output "Already extracted: $sequence"
    } else {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Expand-Archive -LiteralPath $_.FullName -DestinationPath $target -Force
        Write-Output "Expanded ASL sequence: $sequence"
    }
}
