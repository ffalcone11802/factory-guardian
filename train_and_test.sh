#!/bin/bash

python run.py\
    --data_root=data\
    --category=toothbrush\
    --latent_dim=128\
    --batch_size=16\
    --beta=1\
    --num_epochs=200\
    --lr=2e-4\
    --init_type=xavier\
    --save_imgs_freq=10\
    --save_checkpoint_freq=20\
    --verbose=False\
    --num_warm_up_epochs=20\
    --num_inference_epochs=1000\
