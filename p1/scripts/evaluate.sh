#!/bin/bash
set -euo pipefail

uv run p1/inpainting/evaluate.py \
    --data-dir p1/data \
    --models \
        p1/output/part-conv/2026-10-31_11-24-10 \
    --repeats 5 \
    --seed 42 \
    "$@"
