#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile=ui
tests=false
for option in "$@"; do
  case "$option" in
    --gpu) profile=gpu ;;
    --ui) profile=ui ;;
    --test) tests=true ;;
    --help|-h) echo 'Usage: bash scripts/setup.sh [--ui | --gpu] [--test]'; exit 0 ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
done
bootstrap_python="${ANNOLOOP_BOOTSTRAP_PYTHON:-python3.10}"
if ! command -v "$bootstrap_python" >/dev/null; then
  echo 'Cần Python 3.10 và module venv. Xem README.md / docs/INSTALL.md.' >&2; exit 1
fi
"$bootstrap_python" -c 'import sys; assert sys.version_info[:2] == (3,10), "Profile này cần Python 3.10"'
if [[ "$profile" == gpu ]]; then
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'GPU profile cần Linux x86_64' >&2; exit 1; }
  export CXX="${CXX:-g++}"
  export CC="${CC:-gcc}"
  command -v "$CXX" >/dev/null || { echo 'Cần g++: xem docs/INSTALL.md' >&2; exit 1; }
  # Image builds have no GPU: ANNOLOOP_BUILD_NO_GPU=1 needs an explicit TORCH_CUDA_ARCH_LIST.
  if [[ "${ANNOLOOP_BUILD_NO_GPU:-0}" == 1 ]]; then
    [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]] || { echo 'Build không GPU cần TORCH_CUDA_ARCH_LIST' >&2; exit 1; }
  else
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  fi
  compiler_major="$("$CXX" -dumpversion | cut -d. -f1)"
  [[ "$compiler_major" -ge 6 && "$compiler_major" -le 11 ]] || { echo 'CUDA 11.8 profile cần GCC/G++ 6–11; xem docs/INSTALL.md' >&2; exit 1; }
  "$bootstrap_python" -c 'import os,sysconfig; from pathlib import Path; roots=[sysconfig.get_path("include"),*os.getenv("CPATH", "").split(os.pathsep)]; assert any(p and (Path(p)/"Python.h").is_file() for p in roots), "Cần Python headers: cài python3.10-dev (xem docs/INSTALL.md)"'
fi
# Never install/downgrade the caller's active venv.
"$bootstrap_python" "$project_dir/scripts/create_venv.py" "$project_dir/.venv"
ui_python="$project_dir/.venv/bin/python"
"$ui_python" -m pip install 'pip==25.3' 'setuptools==75.8.0' 'wheel==0.45.1'
"$ui_python" -m pip install --constraint "$project_dir/constraints-ui.txt" -r "$project_dir/requirements-ui.txt"
deps_only="${ANNOLOOP_SETUP_DEPS_ONLY:-0}"
[[ "$deps_only" == 1 ]] || "$ui_python" -m pip install --no-deps --no-build-isolation -e "$project_dir"
if [[ "$tests" == true ]]; then
  # CPU Torch is only used by optimizer-state unit tests, never for detection.
  "$ui_python" -m pip install --constraint "$project_dir/constraints-ui.txt" 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
fi
if [[ "$profile" == gpu ]]; then
  "$bootstrap_python" "$project_dir/scripts/create_venv.py" "$project_dir/.venv-gpu"
  model_python="$project_dir/.venv-gpu/bin/python"
  "$model_python" -m pip install 'pip==25.3' 'setuptools==75.8.0' 'wheel==0.45.1'
  "$model_python" -m pip install --constraint "$project_dir/constraints-gpu.txt" 'torch==2.0.0+cu118' --index-url https://download.pytorch.org/whl/cu118
  "$model_python" -m pip install --constraint "$project_dir/constraints-gpu.txt" -r "$project_dir/requirements-gpu.txt"
  [[ "$deps_only" == 1 ]] || "$model_python" -m pip install --no-deps --no-build-isolation -e "$project_dir"
  source "$project_dir/scripts/env.sh"
  toolkit="$ANNOLOOP_CUDA_TOOLKIT"
  if [[ ! -x "$toolkit/bin/nvcc" || ! -f "$toolkit/include/cuda/std/cassert" ]]; then
    "$bootstrap_python" "$project_dir/scripts/install_cuda.py" --output "$toolkit"
  fi
  export CUDA_HOME="$toolkit"
  export PATH="$(dirname -- "$model_python"):$toolkit/bin:$PATH"
  export MAX_JOBS="${MAX_JOBS:-2}"
  # Python's default -g creates large Torch template debug tables on small hosts.
  export CFLAGS="${CFLAGS:+$CFLAGS }-g0"
  export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-g0"
  # Read capability without launching a CUDA process, also when training is active.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)}"
  "$bootstrap_python" -c 'import os; caps=os.environ["TORCH_CUDA_ARCH_LIST"].split(";"); assert caps and all(5 <= float(c.replace("+PTX", "")) <= 9 for c in caps), "GPU này cần profile CUDA mới hơn; CUDA 11.8 hỗ trợ capability 5.x–9.0"'
  (
    cd "$project_dir/vendor/ST3D"
    "$model_python" setup.py build_ext --inplace
  )
  [[ "$deps_only" == 1 ]] || "$model_python" -m annoloop.runtime --check "$project_dir/vendor/ST3D"
  if [[ "${ANNOLOOP_SKIP_MMDET:-0}" != 1 ]]; then
    bash "$project_dir/scripts/setup_mmdet3d.sh"
  fi
fi
echo 'Cài đặt hoàn tất. Mở UI: bash scripts/start_ui.sh'
echo 'UI: http://localhost:8502 · Model jobs cần GPU profile và dữ liệu của bạn.'
