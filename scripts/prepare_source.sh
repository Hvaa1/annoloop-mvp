#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$ANNOLOOP_WORKER_PYTHON" "$project_dir/scripts/prepare_source.py" "$@"
