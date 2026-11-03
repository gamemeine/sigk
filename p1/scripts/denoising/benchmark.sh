#!/usr/bin/env bash
set -euo pipefail
python -m p1.denoising.cli benchmark "$@"
