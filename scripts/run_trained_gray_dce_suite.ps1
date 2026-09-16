param(
    [string[]]$Sequences = @("V1_01_easy", "V1_03_difficult", "V2_03_difficult"),
    [double]$ReplaySpeed = 1.0,
    [string]$Model = "",
    [string]$RunLabel = "dual_gray_dce_trained_official",
    [string]$ConfigPath = "",
    [switch]$ForceRerun
)

$ErrorActionPreference = "Stop"
$Work = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $Work ".deps\gray_dce_cleanroom_scale16_best_480x752.onnx"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Trained ONNX model not found: $Model"
}
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $Work "config\irvio_dual_proxy_euroc_no_loop.yaml"
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "VIO config not found: $ConfigPath"
}
if ($Work -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Expected a Windows drive path for the workspace: $Work"
}
$Drive = $Matches[1].ToLowerInvariant()
$Rest = $Matches[2].Replace('\', '/')
$WorkWsl = "/mnt/$Drive/$Rest"
if ($ConfigPath -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Expected a Windows drive path for the VIO config: $ConfigPath"
}
$ConfigDrive = $Matches[1].ToLowerInvariant()
$ConfigRest = $Matches[2].Replace('\', '/')
$ConfigPathWsl = "/mnt/$ConfigDrive/$ConfigRest"

$env:PYTHONPATH = (Join-Path $Work ".deps\onnx_runner") + ";" +
    (Join-Path $Work ".deps\openvino_windows")

foreach ($Sequence in $Sequences) {
    $SequenceRoot = Join-Path $Work "datasets\$Sequence"
    $ResultRoot = Join-Path $Work "results\$Sequence"
    $EnhancedRoot = Join-Path $ResultRoot ("enhanced_" + $RunLabel)
    $EnhancementReport = Join-Path $ResultRoot ("enhanced_" + $RunLabel + ".json")
    $Metrics = Join-Path $ResultRoot ($RunLabel + "_ate.json")
    if ((Test-Path -LiteralPath $Metrics) -and -not $ForceRerun) {
        Write-Host "[$Sequence] result exists; skipping"
        continue
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SequenceRoot "mav0\cam0\data.csv"))) {
        throw "[$Sequence] dataset is incomplete: $SequenceRoot"
    }
    New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null

    Write-Host "[$Sequence] enhancing with trained gray checkpoint"
    $EnhanceArgs = @(
        (Join-Path $Work "scripts\enhance_euroc_onnx.py"),
        $SequenceRoot, $EnhancedRoot, "--model", $Model, "--backend", "openvino",
        "--skip-existing", "--report", $EnhancementReport
    )
    & $Python -u @EnhanceArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] enhancement failed" }

    Write-Host "[$Sequence] running synchronized dual VIO"
    $WslArgs = @(
        "-d", "Ubuntu-20.04", "--", "bash",
        "$WorkWsl/scripts/run_sequence_dual_asl.sh",
        $Sequence, "$WorkWsl/datasets/$Sequence", "$WorkWsl/results/$Sequence/enhanced_$RunLabel",
        $ReplaySpeed, 0, $ConfigPathWsl, $RunLabel
    )
    & wsl @WslArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] dual VIO failed" }

    Write-Host "[$Sequence] evaluating ATE"
    $EvaluateArgs = @(
        (Join-Path $Work "scripts\evaluate_ate.py"),
        "--estimate", (Join-Path $ResultRoot ($RunLabel + ".csv")),
        "--groundtruth", (Join-Path $SequenceRoot "mav0\state_groundtruth_estimate0\data.csv"),
        "--camera-csv", (Join-Path $SequenceRoot "mav0\cam0\data.csv"),
        "--interpolate", "--output", $Metrics
    )
    & $Python @EvaluateArgs
    if ($LASTEXITCODE -ne 0) { throw "[$Sequence] ATE evaluation failed" }
}

& $Python (Join-Path $Work "scripts\aggregate_trained_gray_results.py") --label $RunLabel --sequences $Sequences
if ($LASTEXITCODE -ne 0) { throw "trained gray result aggregation failed" }
Write-Host "Trained gray DCE suite completed."
