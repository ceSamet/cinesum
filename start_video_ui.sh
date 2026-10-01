#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd -P)"
exec "$project_dir/samet/start_video_ui.sh"
