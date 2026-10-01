#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd "$(dirname "$0")" && pwd -P)"
venv_dir="$(cd "$app_dir/.." && pwd -P)/.venv"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "Sanal ortam bulunamadı: $venv_dir"
  exit 1
fi
cd "$app_dir"
exec "$venv_dir/bin/python" -m scene_captioner.web_app
