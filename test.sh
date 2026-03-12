#!/bin/bash

python run.py\
    --inference_mode=True\
    --data_root=data\
    --category=toothbrush\
    --latent_dim=128\
    --batch_size=16\
    --num_warm_up_epochs=20\
    --num_inference_epochs=1000\
