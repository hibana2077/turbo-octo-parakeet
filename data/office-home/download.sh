#!/usr/bin/env bash
set -euo pipefail

URL="https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/OfficeHomeDataset_10072016.zip?download=true"
ZIP_NAME="OfficeHomeDataset_10072016.zip"
OUT_DIR="OfficeHomeDataset_10072016"

# Download into the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading dataset..."
if command -v curl >/dev/null 2>&1; then
    curl -L "$URL" -o "$ZIP_NAME"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$ZIP_NAME" "$URL"
else
    echo "Error: neither curl nor wget is installed." >&2
    exit 1
fi

echo "Unzipping..."
mkdir -p "$OUT_DIR"
unzip -o "$ZIP_NAME" -d "$OUT_DIR"
mv "$OUT_DIR/OfficeHomeDataset_10072016"/* "images/"
rm -rf "$OUT_DIR" "$ZIP_NAME"

echo "Done."