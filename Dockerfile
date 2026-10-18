# AnnoLoop: Streamlit UI + GPU workers (ST3D + MMDetection3D) in one image.
# Runs on the user's own NVIDIA GPU via the NVIDIA Container Toolkit.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev build-essential git curl ca-certificates \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/annoloop
# 1) Heavy, rarely-changing layer: venvs, Torch, spconv, MMDet3D, compiled ST3D ops.
COPY requirements-ui.txt requirements-gpu.txt constraints-ui.txt constraints-gpu.txt pyproject.toml ./
COPY scripts/ scripts/
COPY vendor/ vendor/

# Compile CUDA ops for common GPU generations (Maxwell → Hopper), not just the build host.
ARG TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ARG MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    ANNOLOOP_CUDA_TOOLKIT=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda
RUN ANNOLOOP_BUILD_NO_GPU=1 ANNOLOOP_SETUP_DEPS_ONLY=1 ANNOLOOP_BOOTSTRAP_PYTHON=python3.10 bash scripts/setup.sh --gpu \
    && rm -rf /root/.cache vendor/ST3D/build

# 2) Cheap layer: application code; code changes rebuild only this part.
COPY . .
RUN . scripts/env.sh \
    && "$VINAI_PYTHON" -m pip install --no-deps --no-build-isolation -e . \
    && "$ANNOLOOP_WORKER_PYTHON" -m pip install --no-deps --no-build-isolation -e . \
    && "$ANNOLOOP_WORKER_PYTHON" -m annoloop.runtime --check vendor/ST3D \
    && "$ANNOLOOP_WORKER_PYTHON" -m annoloop.mmdet_adapter --probe

# Projects/jobs persist in /data; datasets are mounted read-only at /datasets.
ENV ANNOLOOP_DATA_DIR=/data \
    ANNOLOOP_BROWSE_ROOT=/datasets \
    ANNOLOOP_BIND=0.0.0.0 \
    ANNOLOOP_PORT=8502
VOLUME ["/data"]
EXPOSE 8502
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:8502/_stcore/health || exit 1
CMD ["bash", "scripts/start_ui.sh", "--server.headless", "true"]
