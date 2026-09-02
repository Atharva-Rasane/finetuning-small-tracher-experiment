#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STATE_DIR="${EXPERIMENT_STATE_DIR:-$SCRIPT_DIR/state}"
VENV_DIR="$SCRIPT_DIR/.venv"
LOG_DIR="$STATE_DIR/logs"
FINISHED_MARKER="$STATE_DIR/EXPERIMENT_FINISHED.json"
MAX_RESTARTS="${EXPERIMENT_MAX_RESTARTS:-5}"
DRIVER_INSTALLER_URL="https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz"
DRIVER_INSTALLER="$STATE_DIR/tmp/cuda_installer.pyz"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$STATE_DIR/tmp" "$STATE_DIR/hf_cache"

if [[ -f "$FINISHED_MARKER" ]]; then
  echo "EXPERIMENT FINISHED"
  exit 0
fi

have_sudo() {
  command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1
}

install_base_packages_if_needed() {
  local missing=0
  for cmd in python3 curl lspci; do
    command -v "$cmd" >/dev/null 2>&1 || missing=1
  done
  if [[ $missing -eq 0 ]]; then
    return 0
  fi
  if ! have_sudo; then
    echo "ERROR: base packages are missing and passwordless sudo is unavailable."
    exit 2
  fi
  echo "Installing base VM packages..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip python3-venv curl ca-certificates pciutils git build-essential
}

ensure_nvidia_driver() {
  # Fast path: driver already works.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
  fi

  install_base_packages_if_needed

  # Installing a driver cannot create a GPU. Fail clearly if the VM has no NVIDIA PCI device.
  if ! lspci | grep -Eqi 'NVIDIA|3D controller.*NVIDIA|VGA compatible controller.*NVIDIA'; then
    echo "ERROR: no NVIDIA GPU is attached to this VM."
    echo "Stop the VM, attach an NVIDIA T4, start it again, then rerun ./run.sh."
    exit 2
  fi

  if ! have_sudo; then
    echo "ERROR: NVIDIA driver is missing and passwordless sudo is unavailable."
    exit 2
  fi

  echo "NVIDIA GPU detected, but the driver is not usable."
  echo "Installing the driver with Google Cloud's official GPU installer..."

  # The Ops Agent can hold NVIDIA libraries open during driver installation.
  sudo systemctl stop google-cloud-ops-agent >/dev/null 2>&1 || true

  curl -fL --retry 5 --retry-delay 2 \
    "$DRIVER_INSTALLER_URL" \
    -o "$DRIVER_INSTALLER"

  sudo python3 "$DRIVER_INSTALLER" install_driver

  # Some images load the driver immediately; others need one reboot after kernel/module changes.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA driver installed successfully."
    sudo systemctl start google-cloud-ops-agent >/dev/null 2>&1 || true
    return 0
  fi

  touch "$STATE_DIR/DRIVER_REBOOT_REQUIRED"
  echo
  echo "NVIDIA driver installation completed, but the kernel module is not active yet."
  echo "A one-time reboot is required. Run:"
  echo
  echo "    sudo reboot"
  echo
  echo "After reconnecting, return to this repository and run ./run.sh again."
  echo "No experiment progress has been lost."
  exit 20
}

install_base_packages_if_needed
ensure_nvidia_driver
rm -f "$STATE_DIR/DRIVER_REBOOT_REQUIRED" 2>/dev/null || true

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed."
  exit 2
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment..."
  if ! python3 -m venv "$VENV_DIR"; then
    if have_sudo; then
      echo "Installing python3-venv..."
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip
      python3 -m venv "$VENV_DIR"
    else
      echo "ERROR: python3 -m venv failed and passwordless sudo is unavailable."
      exit 2
    fi
  fi
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

REQ_HASH="$($PYTHON - <<'PY'
from pathlib import Path
import hashlib
payload = b"torch==2.7.0+cu118\n" + Path("requirements.txt").read_bytes()
print(hashlib.sha256(payload).hexdigest())
PY
)"
ENV_MARKER="$VENV_DIR/.environment.sha256"
OLD_HASH=""
[[ -f "$ENV_MARKER" ]] && OLD_HASH="$(cat "$ENV_MARKER")"

if [[ "$REQ_HASH" != "$OLD_HASH" ]]; then
  echo "Installing pinned Python environment..."
  "$PIP" install --upgrade pip setuptools wheel
  # CUDA 11.8 wheel is intentionally used for broad T4/driver compatibility.
  "$PIP" install --index-url https://download.pytorch.org/whl/cu118 "torch==2.7.0"
  "$PIP" install -r requirements.txt
  echo "$REQ_HASH" > "$ENV_MARKER"
else
  echo "Python environment already installed."
fi

export EXPERIMENT_STATE_DIR="$STATE_DIR"
export HF_HOME="$STATE_DIR/hf_cache"
export TMPDIR="$STATE_DIR/tmp"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Avoid two experiment supervisors writing the same checkpoints.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$STATE_DIR/run.lock"
  if ! flock -n 9; then
    echo "ERROR: another run.sh process is already using $STATE_DIR"
    exit 3
  fi
fi

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA. Check nvidia-smi / driver setup.")
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"PyTorch {torch.__version__} | CUDA runtime {torch.version.cuda}")
print(f"GPU: {name} | {mem:.2f} GiB VRAM")
if "T4" not in name:
    print("WARNING: this experiment is tuned for an NVIDIA T4; continuing anyway.")
PY

attempt=0
while [[ $attempt -lt $MAX_RESTARTS ]]; do
  if [[ -f "$FINISHED_MARKER" ]]; then
    echo "EXPERIMENT FINISHED"
    exit 0
  fi

  attempt=$((attempt + 1))
  LOG_FILE="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-attempt-${attempt}.log"
  echo "Starting/resuming experiment (supervisor attempt $attempt/$MAX_RESTARTS)..."
  echo "Log: $LOG_FILE"

  set +e
  "$PYTHON" -u experiment.py --state-dir "$STATE_DIR" 2>&1 | tee -a "$LOG_FILE"
  code=${PIPESTATUS[0]}
  set -e

  if [[ $code -eq 0 ]]; then
    if [[ -f "$FINISHED_MARKER" ]]; then
      echo "EXPERIMENT FINISHED"
      exit 0
    fi
    echo "ERROR: experiment.py exited successfully but did not write the finished marker."
    exit 4
  fi

  if [[ $code -eq 130 || $code -eq 143 ]]; then
    echo "Experiment interrupted by user/system. Durable checkpoints are preserved."
    exit $code
  fi

  echo "experiment.py exited with code $code."
  if [[ $attempt -lt $MAX_RESTARTS ]]; then
    echo "Restarting in 5 seconds from the latest durable checkpoint..."
    sleep 5
  fi
done

echo "ERROR: experiment failed $MAX_RESTARTS consecutive supervisor attempts."
echo "All durable progress remains in: $STATE_DIR"
exit 1
