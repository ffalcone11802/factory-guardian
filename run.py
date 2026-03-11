import argparse

from factory_guardian import train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--dataroot", type=str, default="./data",
        help="root directory of dataset containing folders for the categories"
    )
    parser.add_argument(
        "--category", type=str, default="toothbrush",
        help="object category to train or test on"
    )

    # Data augmentation
    parser.add_argument(
        "--rotation_range", type=int, default=10,
        help="degree range for random rotations in the preprocessing (set to 0 to disable)"
    )

    # Model hyperparameters
    parser.add_argument(
        "--latent_dim", type=int, default=128,
        help="dimensionality of the latent space"
    )

    # ELBO loss options
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="beta factor for weighting the KL divergence in the ELBO loss"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200,
        help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="learning rate"
    )

    # Misc
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="number of dataloader workers"
    )
    parser.add_argument(
        "--init_type", type=str, default="xavier",
        choices=["normal", "xavier", "kaiming"],
        help="LiteVAE initialization method"
    )
    parser.add_argument(
        "--save_imgs_freq", type=int, default=10,
        help="save sample reconstructions every save_epoch_freq epochs"
    )
    parser.add_argument(
        "--save_checkpoint_freq", type=int, default=20,
        help="save model checkpoints every checkpoint_freq epochs"
    )
    parser.add_argument(
        "--verbose", type=bool, default=True,
        help="enable verbose output during training"
    )

    # Pipeline
    parser.add_argument(
        "--inference_mode", type=bool, default=False,
        help="avoid training and only run inference on the test set"
    )
    parser.add_argument(
        "--onnx", type=bool, default=False,
        help="measure inference time using ONNX"
    )

    args = parser.parse_args()

    if not args.inference_mode:
        train(args)

    test(args)
