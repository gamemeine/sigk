#!/bin/bash
set -euo pipefail

uv run p1/inpainting/data.py \
    --output-dir p1/data \
    --target-res 256 256 \
    --crop-strategy random \
    --seed 42
