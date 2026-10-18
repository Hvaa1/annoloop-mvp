#!/usr/bin/env bash
# Adds the MMDetection3D 1.4 stack to the model venv (.venv-gpu), the same layout the
# verified local runtime uses (ST3D + MMDet3D in one Torch 2.0.0+cu118 environment).
# Explicit step: setup.sh --gpu calls it; never run automatically on app startup.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
model_python="$ANNOLOOP_WORKER_PYTHON"
[[ -x "$model_python" ]] || { echo 'Chưa có .venv-gpu. Chạy: bash scripts/setup.sh --gpu' >&2; exit 1; }
"$model_python" -m pip install 'numpy==1.26.4' 'mmengine==0.10.7'
"$model_python" -m pip install 'mmcv==2.1.0' -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
"$model_python" -m pip install 'mmdet==3.3.0' 'mmdet3d==1.4.0' 'numpy==1.26.4'
[[ "${ANNOLOOP_SETUP_DEPS_ONLY:-0}" == 1 ]] || "$model_python" -m annoloop.mmdet_adapter --probe
