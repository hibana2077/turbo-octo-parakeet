#!/usr/bin/env bash
set -euo pipefail

# Save files next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

URLS=(
    "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt"

    "https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt"

    "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt"

    "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt"

    "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt"

    "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt"
    "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt"
)

for url in "${URLS[@]}"; do
    echo "Downloading: $url"
    wget -c "$url"
done

echo "Done."