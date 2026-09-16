param(
    [string[]]$Sequences = @("V1_01_easy", "V1_03_difficult", "V2_03_difficult"),
    [double]$ReplaySpeed = 1.0,
    [switch]$ForceRerun,
    [string]$RunLabel = "dual_gray_dce_repeat_20260911",
    [string]$OutputPrefix = "dual_branch_proxy_repeatability"
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
    $RepeatEstimate = Join-Path $ResultRoot "$RunLabel.csv"
    $RepeatMetrics = Join-Path $ResultRoot "$RunLabel`_ate.json"
    if ((Test-Path -LiteralPath $RepeatMetrics) -and -not $ForceRerun) {
        Write-Host "[$Sequence] repeat result exists; skipping"
        continue
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SequenceRoot "mav0\cam0\data.csv"))) {
        throw "[$Sequence] dataset is incomplete: $SequenceRoot"
    }
    if (-not (Test-Path -LiteralPath $EnhancedRoot)) {
        throw "[$Sequence] enhanced images are missing: $EnhancedRoot"
    }

    Write-Host "[$Sequence] repeat synchronized dual branch"
    $WslArgs = @(
        "-d", "Ubuntu-20.04", "--", "bash",
        "$WorkWsl/scripts/run_sequence_dual_asl.sh",
        $Sequence, "$WorkWsl/datasets/$Sequence",
        "$WorkWsl/results/$Sequence/enhanced_openvino_full",
        $ReplaySpeed, 0, "$WorkWsl/config/irvio_dual_proxy_euroc_no_loop.yaml",
        $RunLabel
    )
    & wsl @WslArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] repeat dual VINS run failed" }

    Write-Host "[$Sequence] repeat ATE evaluation"
    $EvaluateArgs = @(
        (Join-Path $Work "scripts\evaluate_ate.py"),
        "--estimate", $RepeatEstimate,
        "--groundtruth", (Join-Path $SequenceRoot "mav0\state_groundtruth_estimate0\data.csv"),
        "--camera-csv", (Join-Path $SequenceRoot "mav0\cam0\data.csv"),
        "--interpolate", "--output", $RepeatMetrics
    )
    & $Python @EvaluateArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] repeat ATE evaluation failed" }
}

& $Python (Join-Path $Work "scripts\compare_dual_proxy_repeatability.py") --run-label $RunLabel --output-prefix $OutputPrefix --replay-speed $ReplaySpeed
if ($LASTEXITCODE -ne 0) { throw "Repeatability comparison failed" }
Write-Host "Dual proxy repeatability check completed."
