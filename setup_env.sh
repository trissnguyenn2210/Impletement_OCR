#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/OCR/bin/python"
PADDLEX_DIR="${PROJECT_DIR}/vendor/PaddleX"

if [[ ! -f "${PADDLEX_DIR}/main.py" ]]; then
  git clone --depth 1 --branch release/3.5 \
    https://github.com/PaddlePaddle/PaddleX "${PADDLEX_DIR}"
fi

uv venv --python 3.12 --allow-existing "${PROJECT_DIR}/OCR"
uv pip install --python "${PYTHON}" \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  paddlepaddle-gpu==3.2.1
uv pip install --python "${PYTHON}" pip
uv pip install --python "${PYTHON}" -e "${PADDLEX_DIR}[base,ocr]"
uv pip install --python "${PYTHON}" -r "${PROJECT_DIR}/requirements.txt"

# The training plugins request GUI OpenCV. Headless OpenCV is enough on servers
# and avoids the libGL dependency.
uv pip uninstall --python "${PYTHON}" \
  opencv-contrib-python opencv-python opencv-python-headless || true
uv pip install --python "${PYTHON}" --reinstall --no-deps \
  opencv-contrib-python-headless==4.10.0.84
uv pip install --python "${PYTHON}" numpy==1.26.4 \
  "setuptools<81" visualdl tqdm terminaltables typeguard lapx motmetrics pyclipper
uv pip install --python "${PYTHON}" --no-deps -e "${PADDLEX_DIR}"

# PP-DocLayout training needs PaddleDetection. Disable visible GPUs only while
# registering the plugin so its optional rotated-detection CUDA extension is
# skipped; layout training itself still uses the GPU later.
CUDA_VISIBLE_DEVICES='' PADDLE_PDX_MODEL_SOURCE=bos \
  "${PROJECT_DIR}/OCR/bin/paddlex" \
  --install PaddleDetection --use_local_repos --no_deps

# Reapply the small project-specific training patches after registering the
# local PaddleX/PaddleDetection repositories.
PADDLEX_PATCH="${PROJECT_DIR}/patches/paddlex_object_detection.patch"
PPDET_PATCH="${PROJECT_DIR}/patches/paddledetection_early_stopping.patch"
if git -C "${PADDLEX_DIR}" apply --check "${PADDLEX_PATCH}"; then
  git -C "${PADDLEX_DIR}" apply "${PADDLEX_PATCH}"
fi
PPDET_DIR="${PADDLEX_DIR}/paddlex/repo_manager/repos/PaddleDetection"
if git -C "${PPDET_DIR}" apply --recount --check "${PPDET_PATCH}"; then
  git -C "${PPDET_DIR}" apply --recount "${PPDET_PATCH}"
fi

# PaddleX extras can reinstall GUI OpenCV. Keep only the headless wheel.
uv pip uninstall --python "${PYTHON}" \
  opencv-contrib-python opencv-python opencv-python-headless || true
uv pip install --python "${PYTHON}" --reinstall --no-deps \
  opencv-contrib-python-headless==4.10.0.84

# PaddleDetection requires this patched imgaug build. Install without its
# unpinned GUI-OpenCV dependency; the required runtime dependencies are
# already installed above.
uv pip install --python "${PYTHON}" --no-deps \
  "imgaug @ https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/patched_packages/imgaug-0.4.0%2Bpdx-py2.py3-none-any.whl"
uv pip install --python "${PYTHON}" --reinstall --no-deps \
  numpy==1.26.4 opencv-contrib-python-headless==4.10.0.84

echo "Environment ready. Activate with: source ${PROJECT_DIR}/OCR/bin/activate"
