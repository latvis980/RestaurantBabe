#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Download Chromium and all system dependencies for Playwright
playwright install chromium --with-deps
