#!/bin/bash
set -e  # stop immediately if any command fails

# -----------------------------
# Build script for Render.com
# -----------------------------

echo "Starting build..."

# 1️⃣ Ensure Poetry installs all Python dependencies
poetry install --no-interaction --no-ansi

# 2️⃣ Optional: If you use imageio-ffmpeg, make sure it's installed via Poetry
# This avoids system-level ffmpeg installation, which Render can't do
# poetry add imageio-ffmpeg  # Uncomment if not already in pyproject.toml

echo "Build complete!"
