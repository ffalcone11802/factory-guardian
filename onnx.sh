#!/bin/bash

python run_onnx.py\
    --category=toothbrush\
    --latent_dim=128\
    --num_warm_up_epochs=100\
    --num_inference_epochs=1000\
