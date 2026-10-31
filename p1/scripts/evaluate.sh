#!/bin/bash
set -euo pipefail

uv run p1/inpainting/evaluate.py \
    --data-dir p1/data \
    --seed 42
