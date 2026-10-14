#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$VINAI_PYTHON" -m unittest discover -s "$project_dir/tests" -v "$@"
