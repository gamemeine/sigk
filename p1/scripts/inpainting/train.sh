#!/bin/bash

CONFIG=${1:-"p1/configs/unet_p0_1.yaml"}

uv run -m p1.inpainting.modeling.unet.train \
    --config "${CONFIG}"
