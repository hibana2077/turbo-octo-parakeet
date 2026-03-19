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

# Unzip all downloaded zip files in this directory
for zip_file in ./*.zip; do
    [ -e "$zip_file" ] || continue
    echo "Unzipping: $zip_file"
    unzip -o -q "$zip_file"
done

# Remove zip files after extraction
rm -f ./*.zip

# Add the dataset prefix to the txt files
sed -i 's#^#data/DomainNet/clipart/#' clipart_train.txt
sed -i 's#^#data/DomainNet/clipart/#' clipart_test.txt
sed -i 's#^#data/DomainNet/infograph/#' infograph_train.txt
sed -i 's#^#data/DomainNet/infograph/#' infograph_test.txt
sed -i 's#^#data/DomainNet/painting/#' painting_train.txt
sed -i 's#^#data/DomainNet/painting/#' painting_test.txt
sed -i 's#^#data/DomainNet/quickdraw/#' quickdraw_train.txt
sed -i 's#^#data/DomainNet/quickdraw/#' quickdraw_test.txt
sed -i 's#^#data/DomainNet/real/#' real_train.txt
sed -i 's#^#data/DomainNet/real/#' real_test.txt
sed -i 's#^#data/DomainNet/sketch/#' sketch_train.txt
sed -i 's#^#data/DomainNet/sketch/#' sketch_test.txt

echo "Done."