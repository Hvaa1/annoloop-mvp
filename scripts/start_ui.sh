#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ ! -x "$VINAI_PYTHON" ]]; then
  echo 'Chưa có môi trường UI. Chạy: bash scripts/setup.sh' >&2
  exit 1
fi
exec "$VINAI_PYTHON" -m streamlit run "$project_dir/ui/app.py" --server.address "${ANNOLOOP_BIND:-127.0.0.1}" --server.port "${ANNOLOOP_PORT:-8502}" --server.fileWatcherType "${ANNOLOOP_FILE_WATCHER:-poll}" --theme.base light --theme.primaryColor '#7054d8' --theme.backgroundColor '#f7f8fc' --theme.secondaryBackgroundColor '#ffffff' --theme.textColor '#222039' "$@"
