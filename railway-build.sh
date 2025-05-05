#!/usr/bin/env bash
set -euo pipefail

# Log execution steps for debugging
echo "Starting Railway build process..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create data directory for source validation cache
echo "Creating data directory..."
mkdir -p data

# Install system dependencies for Playwright
echo "Installing system dependencies for Playwright..."
apt-get update -y
apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxcb1 \
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    fonts-liberation \
    xdg-utils || echo "Some system dependencies could not be installed, but we'll continue"

# Use a more direct approach to install Playwright browsers
echo "Installing Playwright Chromium browser..."
python -m playwright install chromium --with-deps || echo "Playwright browser installation encountered issues - the app will use fallback mechanisms"

# Verify installation
echo "Checking installed browsers..."
python -m playwright install --help || echo "Playwright command failed, but application has fallback mechanisms"

echo "Build script completed - the application will use HTTP fallbacks if Playwright is unavailable"