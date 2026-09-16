$ErrorActionPreference = "Stop"

$Workspace = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source
$OpenCvTarget = Join-Path $Workspace ".deps\onnx_runner"
$OpenVinoTarget = Join-Path $Workspace ".deps\openvino_windows"

& $Python -m pip install --target $OpenCvTarget "opencv-python-headless==4.10.0.84"
& $Python -m pip install --target $OpenVinoTarget "openvino==2024.6.0"

Write-Host "Fast inference dependencies installed inside $Workspace\.deps"
Write-Host "Set PYTHONPATH to both target directories before running enhance_euroc_onnx.py."
