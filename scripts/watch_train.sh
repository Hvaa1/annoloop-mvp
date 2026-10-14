#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$VINAI_PYTHON" "$project_dir/scripts/training_monitor.py" "$@"
