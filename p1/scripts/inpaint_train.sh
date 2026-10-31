#!/bin/bash
set -euo pipefail

source .venv/bin/activate

python -m p1.inpainting.modeling.pconv.train \
    --config p1/configs/pconv_original.yaml \
    --evaluate \
    "$@"
