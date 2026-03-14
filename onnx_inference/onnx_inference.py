from argparse import Namespace
import onnxruntime as ort
import torch
from torch import nn, Tensor

from factory_guardian import inference_time
from factory_guardian.model import LiteVAE
from factory_guardian.utils.folder import check_folder, path_joiner, WEIGHTS_FOLDER, ONNX_FOLDER
from factory_guardian.utils.seed import set_seed


def onnx_inference(args: Namespace):
    """
    Main function to perform inference using ONNX Runtime.

    Args:
        args (Namespace): Command-line arguments.
    """
    # Set seed
    set_seed(args.seed)

    # Dummy input
    img_size = 256
    dummy_input = torch.randn(1, 3, img_size, img_size)

    category = args.category

    # Model
    model = LiteVAE(latent_dim=args.latent_dim)

    # Load weights
    model_path = path_joiner(WEIGHTS_FOLDER, f"{category}.pth")
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model parameters loaded from {str(model_path)}")
    print("--------------------------------------------------------")

    model.eval()
    model.cpu()

    # If the ONNX model doesn't already exist, export it'
    check_folder(ONNX_FOLDER)
    onnx_path = path_joiner(ONNX_FOLDER, f"{category}_model.onnx")

    if not onnx_path.exists():
        onnx_export(model, category, dummy_input)

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Define a wrapper function for ONNX Runtime inference
    # to make it have the same input/output format as the PyTorch model
    def onnx_session_wrapper(x):
        return session.run(None, {"input": x})

    # Inference time and FPS
    ms, fps = inference_time(
        model=onnx_session_wrapper,
        x=dummy_input.cpu().numpy(),
        warm_up_epochs=args.num_warm_up_epochs,
        N=args.num_inference_epochs
    )
    print(f"ONNX Runtime - Average Inference Time: {ms:.4f} ms, FPS: {fps:.4f}")


def onnx_export(model: nn.Module, category: str, dummy_input: Tensor):
    """
    Export a PyTorch model to ONNX format.

    Export the provided PyTorch model into an ONNX file suitable for inference.
    The file is saved in a predefined folder with a name based on the category.

    Args:
        model (nn.Module): PyTorch model to be exported.
        category (str): Category name used for naming the exported file.
        dummy_input (Tensor): Dummy input tensor for the model.
    """
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=path_joiner(ONNX_FOLDER, f"{category}_model.onnx"),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        strict=False
    )
