#!/bin/bash
set -euo pipefail

MODEL_URL="https://github.com/hibana2077/turbo-octo-parakeet/releases/download/v1.0.0-pt/imagenet21k_ViT-B_16.npz"

# start download
echo "Downloading pretrained model from ${MODEL_URL}..."
curl -L -o imagenet21k_ViT-B_16.npz "${MODEL_URL}"
echo "Download completed and saved to imagenet21k_ViT-B_16.npz"