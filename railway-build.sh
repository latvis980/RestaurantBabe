#!/usr/bin/env bash
set -euo pipefail

echo "Installing Python deps…"
pip install --no-cache-dir -r requirements.txt

echo "Downloading Chromium and its system libs…"
python -m playwright install --with-deps chromium
