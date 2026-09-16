param(
    [string]$Sequence = "V1_03_difficult",
    [double[]]$AlphaValues = @(2.0, 4.0, 8.0),
    [double]$ReplaySpeed = 1.0,
    [switch]$ForceRerun,
    [string]$OutputPrefix = "v1_03_alpha_sensitivity_20260911"
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

$SequenceRoot = Join-Path $Work "datasets\$Sequence"
$ResultRoot = Join-Path $Work "results\$Sequence"
$EnhancedRoot = Join-Path $ResultRoot "enhanced_openvino_full"
$CanonicalConfig = Join-Path $Work "config\irvio_dual_proxy_euroc_no_loop.yaml"
$TempRoot = Join-Path $Work "tmp\dual_proxy_sensitivity"
New-Item -ItemType Directory -Force -Path $ResultRoot, $TempRoot | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $SequenceRoot "mav0\cam0\data.csv"))) {
    throw "[$Sequence] dataset is incomplete: $SequenceRoot"
}
if (-not (Test-Path -LiteralPath $EnhancedRoot)) {
    throw "[$Sequence] enhanced images are missing: $EnhancedRoot"
}

$env:PYTHONPATH = (Join-Path $Work ".deps\onnx_runner") + ";" +
    (Join-Path $Work ".deps\openvino_windows")

$rows = @()
$canonicalText = Get-Content -LiteralPath $CanonicalConfig -Raw
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$canonicalDual = Import-Csv -LiteralPath (Join-Path $Work "results\dual_branch_proxy_summary.csv") |
    Where-Object { $_.sequence -eq $Sequence } | Select-Object -First 1
if (-not $canonicalDual) {
    throw "Canonical dual summary has no row for $Sequence"
}

foreach ($alpha in $AlphaValues) {
    $alphaText = $alpha.ToString("0.###", $invariant)
    $alphaTag = "a" + $alphaText.Replace('.', 'p')
    $configName = "irvio_dual_proxy_$alphaTag.yaml"
    $configPath = Join-Path $TempRoot $configName
    $runLabel = "${OutputPrefix}_$alphaTag"
    $estimate = Join-Path $ResultRoot "$runLabel.csv"
    $metrics = Join-Path $ResultRoot "${runLabel}_ate.json"

    if (-not (Test-Path -LiteralPath $configPath) -or $ForceRerun) {
        $variantText = [regex]::Replace(
            $canonicalText,
            '(?m)^adaptive_delta_alpha:\s*[-+0-9.eE]+\s*$',
            "adaptive_delta_alpha: $alphaText"
        )
        if ($variantText -eq $canonicalText) {
            throw "Could not replace adaptive_delta_alpha in $CanonicalConfig"
        }
        [System.IO.File]::WriteAllText(
            $configPath,
            $variantText,
            (New-Object System.Text.UTF8Encoding($false))
        )
    }

    if ((Test-Path -LiteralPath $metrics) -and -not $ForceRerun) {
        Write-Host "[$Sequence][$alphaText] metric exists; skipping replay"
    } else {
        Write-Host "[$Sequence][$alphaText] running synchronized dual branch"
        $configWsl = "$WorkWsl/tmp/dual_proxy_sensitivity/$configName"
        $WslArgs = @(
            "-d", "Ubuntu-20.04", "--", "bash",
            "$WorkWsl/scripts/run_sequence_dual_asl.sh",
            $Sequence, "$WorkWsl/datasets/$Sequence",
            "$WorkWsl/results/$Sequence/enhanced_openvino_full",
            $ReplaySpeed, 0, $configWsl, $runLabel
        )
        & wsl @WslArgs
        if ($LASTEXITCODE -ne 0) { throw "[$Sequence][$alphaText] dual VINS run failed" }

        Write-Host "[$Sequence][$alphaText] evaluating ATE"
        $EvaluateArgs = @(
            (Join-Path $Work "scripts\evaluate_ate.py"),
            "--estimate", $estimate,
            "--groundtruth", (Join-Path $SequenceRoot "mav0\state_groundtruth_estimate0\data.csv"),
            "--camera-csv", (Join-Path $SequenceRoot "mav0\cam0\data.csv"),
            "--interpolate", "--output", $metrics
        )
        & $Python @EvaluateArgs
        if ($LASTEXITCODE -ne 0) { throw "[$Sequence][$alphaText] ATE evaluation failed" }
    }

    $metric = Get-Content -LiteralPath $metrics -Raw | ConvertFrom-Json
    $rows += [pscustomobject]@{
        sequence = $Sequence
        adaptive_delta_alpha = [double]$alpha
        replay_speed = [double]$ReplaySpeed
        run_label = $runLabel
        estimate_poses = [int]$metric.estimate_poses
        matched_poses = [int]$metric.matched_poses
        time_coverage = [double]$metric.time_coverage
        ate_rmse_se3_m = [double]$metric.ate_rmse_se3_m
        ate_rmse_sim3_m = [double]$metric.ate_rmse_sim3_m
        sim3_scale = [double]$metric.sim3_scale
        baseline_ate_rmse_se3_m = [double]$canonicalDual.baseline_ate_rmse_se3_m
        weighting_ate_rmse_se3_m = [double]$canonicalDual.weighting_ate_rmse_se3_m
        vs_baseline_reduction_percent = (([double]$canonicalDual.baseline_ate_rmse_se3_m - [double]$metric.ate_rmse_se3_m) / [double]$canonicalDual.baseline_ate_rmse_se3_m * 100.0)
        vs_weighting_reduction_percent = (([double]$canonicalDual.weighting_ate_rmse_se3_m - [double]$metric.ate_rmse_se3_m) / [double]$canonicalDual.weighting_ate_rmse_se3_m * 100.0)
        pipeline_success = [bool]$metric.pipeline_success
        metric_pass_ate_lt_1m = [bool]$metric.metric_pass_ate_lt_1m
    }
}

$outputCsv = Join-Path $Work "results\$OutputPrefix.csv"
$outputJson = Join-Path $Work "results\$OutputPrefix.json"
$rows | Export-Csv -LiteralPath $outputCsv -NoTypeInformation -Encoding UTF8
$document = [pscustomobject]@{
    status = "clean_room_proxy_alpha_sensitivity"
    sequence = $Sequence
    canonical_config = $CanonicalConfig
    parameter = "adaptive_delta_alpha"
    fixed_parameters = "adaptive_delta_beta=4.0, source_separation=1, grayscale_DCE_OpenVINO"
    replay_speed = [double]$ReplaySpeed
    results = @($rows)
    warning = "This is a clean-room proxy sensitivity study; it does not recover unpublished author parameters or validate official IR-VIO."
}
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outputJson -Encoding UTF8
Write-Host "Saved $outputCsv"
Write-Host "Saved $outputJson"
