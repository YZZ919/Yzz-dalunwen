#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_torch=true
install_ros=false
check_only=false

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_ubuntu_4090.sh [options]

Options:
  --install-ros  Install ROS Noetic build/runtime dependencies (Ubuntu 20.04).
  --skip-torch  Do not install the project-local CUDA PyTorch environment.
  --check-only  Run validation without installing anything.
  -h, --help    Show this help.

The default action installs only project-local PyTorch under .deps/torch_cuda.
It does not download datasets, checkpoints, or configure Git remotes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-ros) install_ros=true ;;
    --skip-torch) install_torch=false ;;
    --check-only) check_only=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "${check_only}" == true ]]; then
  exec bash "${workspace}/scripts/check_ubuntu_4090.sh"
fi

if [[ "${install_ros}" == true ]]; then
  if [[ ! -r /etc/os-release ]]; then
    printf 'Cannot verify the operating system for ROS Noetic.\n' >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "20.04" ]]; then
    printf 'ROS Noetic automated setup requires Ubuntu 20.04; found %s %s.\n' "${ID:-unknown}" "${VERSION_ID:-unknown}" >&2
    exit 2
  fi
  if [[ "${EUID}" -eq 0 ]]; then
    bash "${workspace}/scripts/bootstrap_wsl_noetic.sh"
  elif command -v sudo >/dev/null 2>&1; then
    sudo bash "${workspace}/scripts/bootstrap_wsl_noetic.sh"
  else
    printf 'sudo is required for ROS dependency installation.\n' >&2
    exit 2
  fi
fi

if [[ "${install_torch}" == true ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is unavailable. Install the NVIDIA driver before PyTorch.\n' >&2
    exit 2
  fi
  bash "${workspace}/scripts/bootstrap_gray_dce_cuda.sh"
fi

bash "${workspace}/scripts/check_ubuntu_4090.sh"

