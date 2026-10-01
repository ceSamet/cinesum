#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Önce kurulum yap: python3 -m venv .venv && .venv/bin/pip install -r requirements-models.txt"
  exit 1
fi
exec .venv/bin/python -m scene_captioner.web_app
