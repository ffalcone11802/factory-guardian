import argparse

from onnx_inference import onnx_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--category", type=str, default="toothbrush",
        help="object category to run inference on "
             "(only for compatibility with evaluation done using the PyTorch model)"
    )

    # Model hyperparameters
    parser.add_argument(
        "--latent_dim", type=int, default=128,
        help="dimensionality of the latent space"
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed for reproducibility"
    )

    # Inference params
    parser.add_argument(
        "--num_warm_up_epochs", type=int, default=100,
        help="number of epochs to warm up the model"
    )
    parser.add_argument(
        "--num_inference_epochs", type=int, default=1000,
        help="number of epochs to run inference on a dummy input"
    )

    args = parser.parse_args()

    onnx_inference(args)
