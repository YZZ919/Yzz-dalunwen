param(
    [string[]]$Sequences = @(
        "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
        "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
        "V2_01_easy", "V2_02_medium", "V2_03_difficult"
    ),
    # Real-time replay avoids queue back-pressure and is the safer default for
    # repeatable VINS initialization; pass -ReplaySpeed 3 for a faster run.
    [double]$ReplaySpeed = 1.0,
    [switch]$ForceRerun
)

$ErrorActionPreference = "Stop"
$Work = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source
if ($Work -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Expected a Windows drive path for the workspace: $Work"
}
$Drive = $Matches[1].ToLowerInvariant()
$Rest = $Matches[2].Replace('\', '/')
$WorkWsl = "/mnt/$Drive/$Rest"

$env:PYTHONPATH = (Join-Path $Work ".deps\onnx_runner") + ";" +
    (Join-Path $Work ".deps\openvino_windows")

foreach ($Sequence in $Sequences) {
    $SequenceRoot = Join-Path $Work "datasets\$Sequence"
    $ResultRoot = Join-Path $Work "results\$Sequence"
    $EnhancedRoot = Join-Path $ResultRoot "enhanced_openvino_full"
    $EnhancementReport = Join-Path $ResultRoot "enhanced_openvino_full.json"
    $Metrics = Join-Path $ResultRoot "dual_gray_dce_full_ate.json"
    if ((Test-Path -LiteralPath $Metrics) -and -not $ForceRerun) {
        Write-Host "[$Sequence] complete result exists; skipping"
        continue
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SequenceRoot "mav0\cam0\data.csv"))) {
        throw "[$Sequence] dataset is incomplete: $SequenceRoot"
    }
    New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null

    Write-Host "[$Sequence] enhancing images"
    $EnhanceArgs = @(
        (Join-Path $Work "scripts\enhance_euroc_onnx.py"),
        $SequenceRoot, $EnhancedRoot, "--backend", "openvino",
        "--skip-existing", "--report", $EnhancementReport
    )
    & $Python -u @EnhanceArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] enhancement failed" }

    Write-Host "[$Sequence] running synchronized dual branch in WSL"
    $WslArgs = @(
        "-d", "Ubuntu-20.04", "--", "bash",
        "$WorkWsl/scripts/run_sequence_dual_asl.sh",
        $Sequence, "$WorkWsl/datasets/$Sequence",
        "$WorkWsl/results/$Sequence/enhanced_openvino_full",
        $ReplaySpeed, 0, "$WorkWsl/config/irvio_dual_proxy_euroc_no_loop.yaml",
        "dual_gray_dce_full"
    )
    & wsl @WslArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] dual VINS run failed" }

    Write-Host "[$Sequence] evaluating ATE"
    $EvaluateArgs = @(
        (Join-Path $Work "scripts\evaluate_ate.py"),
        "--estimate", (Join-Path $ResultRoot "dual_gray_dce_full.csv"),
        "--groundtruth", (Join-Path $SequenceRoot "mav0\state_groundtruth_estimate0\data.csv"),
        "--camera-csv", (Join-Path $SequenceRoot "mav0\cam0\data.csv"),
        "--interpolate", "--output", $Metrics
    )
    & $Python @EvaluateArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] ATE evaluation failed" }
}

& $Python (Join-Path $Work "scripts\aggregate_dual_proxy_results.py")
if ($LASTEXITCODE -ne 0) { throw "Dual proxy aggregation failed" }
Write-Host "Dual proxy suite completed."
