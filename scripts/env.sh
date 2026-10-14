#!/usr/bin/env bash
# Source this from entry points. Paths remain relative to this checkout.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$project_dir/.env" ]]; then
  set -a
  source "$project_dir/.env"
  set +a
fi
export ANNOLOOP_ROOT="${ANNOLOOP_ROOT:-$project_dir}"
export VINAI_PYTHON="${VINAI_PYTHON:-$project_dir/.venv/bin/python}"
export ANNOLOOP_WORKER_PYTHON="${ANNOLOOP_WORKER_PYTHON:-$project_dir/.venv-gpu/bin/python}"
export ANNOLOOP_ST3D_RUNTIME="${ANNOLOOP_ST3D_RUNTIME:-$project_dir/vendor/ST3D}"
export ANNOLOOP_CUDA_TOOLKIT="${ANNOLOOP_CUDA_TOOLKIT:-$project_dir/runtime/cuda-11.8}"
export CUDA_HOME="${CUDA_HOME:-$ANNOLOOP_CUDA_TOOLKIT}"
export ANNOLOOP_DATA_DIR="${ANNOLOOP_DATA_DIR:-$project_dir/data}"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
