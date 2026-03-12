import torch
import onnxruntime as ort

from factory_guardian import inference_time
from factory_guardian.model import LiteVAE
from factory_guardian.utils.folder import path_joiner, WEIGHTS_FOLDER, ONNX_FOLDER, check_dir

img_size = 256
dummy_input = torch.randn(1, 3, img_size, img_size)


def onnx_inference(args):
    model = LiteVAE(latent_dim=args.latent_dim)

    category = args.category

    # Load weights
    model_path = path_joiner(WEIGHTS_FOLDER, f"{category}.pth")
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model parameters loaded from {str(model_path)}")
    print("--------------------------------------------------------")

    model.eval()
    model.cpu()

    check_dir(ONNX_FOLDER)
    onnx_path = path_joiner(ONNX_FOLDER, f"{category}_model.onnx")

    if not onnx_path.exists():
        onnx_export(model, category)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    x_np = dummy_input.cpu().numpy()
    session.run(None, {"input": x_np})

    def onnx_session_wrapper(x):
        return session.run(None, {"input": x})

    ms, fps = inference_time(
        onnx_session_wrapper,
        x_np,
        warmup_epochs=args.num_warm_up_epochs,
        N=args.num_inference_epochs
    )
    print(f"ONNX Runtime - Average Inference Time: {ms:.4f} ms, FPS: {fps:.4f}")


def onnx_export(model, category):
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
