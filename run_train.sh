#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/OCR/bin/python"

usage() {
  cat <<'EOF'
Usage:
  ./run_train.sh --list-mig
  ./run_train.sh --mig 0 [train.py options]
  ./run_train.sh --mig MIG-xxxxxxxx [train.py options]

Examples:
  ./run_train.sh --mig 0
  ./run_train.sh --mig 1 --epochs 80
EOF
}

list_mig_devices() {
  nvidia-smi -L | sed -n 's/.*MIG .*UUID: \(MIG-[^)]*\).*/\1/p' | nl -v 0 -w 1 -s ': '
}

resolve_mig_uuid() {
  local selection="$1"
  local -a mig_uuids
  mapfile -t mig_uuids < <(nvidia-smi -L | sed -n 's/.*MIG .*UUID: \(MIG-[^)]*\).*/\1/p')

  if ((${#mig_uuids[@]} == 0)); then
    echo "No MIG device found." >&2
    exit 1
  fi

  if [[ "$selection" =~ ^[0-9]+$ ]]; then
    if ((selection >= ${#mig_uuids[@]})); then
      echo "Invalid MIG index: ${selection}" >&2
      list_mig_devices >&2
      exit 1
    fi
    printf '%s\n' "${mig_uuids[$selection]}"
    return
  fi

  for uuid in "${mig_uuids[@]}"; do
    if [[ "$uuid" == "$selection" ]]; then
      printf '%s\n' "$uuid"
      return
    fi
  done

  echo "MIG UUID is not available: ${selection}" >&2
  list_mig_devices >&2
  exit 1
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found." >&2
  exit 1
fi

if [[ "${1:-}" == "--list-mig" ]]; then
  list_mig_devices
  exit 0
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -eq 0 ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--mig" || $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

MIG_UUID="$(resolve_mig_uuid "$2")"
shift 2

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

echo "Using MIG device: ${MIG_UUID}"
echo "Paddle device: gpu:0"

CUDA_VISIBLE_DEVICES="$MIG_UUID" \
  "$PYTHON" "${PROJECT_DIR}/train.py" --mode train \
  --device gpu:0 \
  "$@"
