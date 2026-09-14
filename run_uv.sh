#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${PROJECT_DIR}/OCR"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Python environment not found: ${ENV_DIR}" >&2
  echo "Run ./setup_env.sh first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
UV_PROJECT_ENVIRONMENT="${ENV_DIR}" \
  uv run --no-sync python train.py "$@"
