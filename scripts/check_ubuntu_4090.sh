#!/usr/bin/env bash
set -uo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
errors=0
warnings=0

ok() { printf '[OK]   %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; errors=$((errors + 1)); }

printf 'IR-VIO 4090 workstation check\n'
printf 'Workspace: %s\n\n' "${workspace}"

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  ok "OS: ${PRETTY_NAME:-unknown}"
  if [[ "${ID:-}" != "ubuntu" ]]; then
    warn "ROS Noetic setup is written for Ubuntu."
  elif [[ "${VERSION_ID:-}" != "20.04" ]]; then
    warn "ROS Noetic is natively targeted at Ubuntu 20.04; current version is ${VERSION_ID:-unknown}."
  fi
else
  warn "Cannot read /etc/os-release."
fi

if command -v git >/dev/null 2>&1; then
  ok "$(git --version)"
else
  fail "git is not installed."
fi

if command -v python3 >/dev/null 2>&1; then
  ok "$(python3 --version 2>&1)"
else
  fail "python3 is not installed."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_info="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n 1)"
  if [[ -n "${gpu_info}" ]]; then
    ok "GPU: ${gpu_info}"
    if [[ "${gpu_info}" != *"4090"* ]]; then
      warn "The first visible GPU is not reported as an RTX 4090."
    fi
  else
    fail "nvidia-smi exists but did not return GPU information."
  fi
else
  fail "nvidia-smi is unavailable; install a compatible NVIDIA driver first."
fi

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  ok "ROS Noetic found at /opt/ros/noetic."
else
  warn "ROS Noetic is not installed; training can run, but VIO build/evaluation cannot."
fi

torch_path="${workspace}/.deps/torch_cuda"
if [[ -d "${torch_path}" ]] && command -v python3 >/dev/null 2>&1; then
  if PYTHONPATH="${torch_path}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
import torch
print("[OK]   PyTorch: version=%s cuda=%s device=%s" % (
    torch.__version__,
    torch.version.cuda,
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
))
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    :
  else
    fail "Project-local PyTorch exists but CUDA validation failed."
  fi
else
  warn "Project-local CUDA PyTorch is not installed; run scripts/setup_ubuntu_4090.sh."
fi

if [[ -d "${workspace}/datasets/sice_gray" ]]; then
  ok "Training dataset directory exists: datasets/sice_gray"
else
  warn "datasets/sice_gray is absent; restore datasets outside Git before training."
fi

if [[ -d "${workspace}/src/VINS-Mono/.git" ]]; then
  ok "VINS-Mono nested repository exists."
  if [[ -n "$(git -C "${workspace}/src/VINS-Mono" status --short 2>/dev/null)" ]]; then
    warn "VINS-Mono has uncommitted changes; record them before a formal run."
  fi
else
  warn "src/VINS-Mono is absent or is not an independent Git checkout."
fi

if [[ -d "${workspace}/src/Noise-AwareCameraExposureControl/.git" ]]; then
  ok "Noise-AwareCameraExposureControl nested repository exists."
else
  warn "src/Noise-AwareCameraExposureControl is absent or is not an independent Git checkout."
fi

if git -C "${workspace}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  project_commit="$(git -C "${workspace}" rev-parse --short HEAD 2>/dev/null || true)"
  ok "Project Git repository found${project_commit:+ at ${project_commit}}."
  if [[ -z "$(git -C "${workspace}" remote 2>/dev/null)" ]]; then
    warn "The project repository has no remote yet."
  fi
else
  fail "The project root is not a Git repository."
fi

df -h "${workspace}" | awk 'NR == 1 || NR == 2 {print "[INFO] " $0}'
printf '\nSummary: %d error(s), %d warning(s).\n' "${errors}" "${warnings}"
exit "${errors}"

