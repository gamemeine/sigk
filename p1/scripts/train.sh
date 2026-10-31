#!/bin/bash
set -euo pipefail

uv run -m p1.inpainting.modeling.pconv.train \
    "$@"
