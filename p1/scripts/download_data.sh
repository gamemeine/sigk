#!/bin/bash
set -euo pipefail

mkdir -p p1/data/raw/DIV2K
cd p1/data/raw/DIV2K

for split in train valid; do
    wget -nc "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_${split}_HR.zip"
    unzip -q -n -j "DIV2K_${split}_HR.zip" '*.png' -d "$split"
done
