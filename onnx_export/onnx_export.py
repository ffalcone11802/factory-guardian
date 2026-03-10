import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from loss import apply_gaussian_smoothing
from model_new import LiteVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiteVAE().to(device)
category = "toothbrush"
num_epochs = 200

with open(os.path.join("..", "weights", f"{category}.json"), "r") as f:
    config = json.load(f)
threshold = config["px_level_th"]
th = config["img_level_th"]


# class LiteVAEExportWrapper(nn.Module):
#     def __init__(self, m):
#         super().__init__()
#         self.model = m
#
#     def forward(self, x):
#         h = self.model.enc(x)
#         h = torch.flatten(h, 1)
#
#         mu = self.model.fc_mu(h)
#         logvar = self.model.fc_logvar(h)
#
#         z = mu
#         recon = self.model.decode(z)
#
#         return recon

# model_.eval()
# model = LiteVAEExportWrapper(model_)
model.load_state_dict(torch.load(os.path.join("..", "weights", f"{category}.pth"), weights_only=True))
model.eval()
model.cpu()


# 1️⃣ Esporta il modello
dummy_input = torch.randn(1, 3, 256, 256)
# torch.onnx.export(model, dummy_input, "model.onnx",
#                   export_params=True,
#                   opset_version=17,
#                   input_names=["input"],
#                   output_names=["output"],
#                   dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
#                   strict=False)


# 2️⃣ Crea sessione ONNX Runtime
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

# 3️⃣ Inferenza
x_np = dummy_input.cpu().numpy()
outputs = session.run(None, {"input": x_np})

# Warm-up
for _ in range(100):
    session.run(None, {"input": x_np})

# Misura inferenza
N = 1000

start = time.perf_counter()
for _ in range(N):
    outputs = session.run(None, {"input": x_np})

    outputs = torch.tensor(outputs[0])
    inputs = torch.tensor(x_np)
    error_map = torch.sum((inputs - outputs) ** 2, dim=1, keepdim=True)
    error_map = apply_gaussian_smoothing(error_map, 3, 2.0)

    thresholded = (error_map > threshold).flatten().cpu().numpy().astype(int)

    flat = error_map.view(error_map.size(0), -1)
    k = max(1, int(0.01 * flat.size(1)))
    topk = torch.topk(flat, k, dim=1)[0]
    score = topk.mean(dim=1)

    pred = (score > th).cpu().numpy().astype(int)
end = time.perf_counter()

avg_time_ms = (end - start) / N * 1000
fps = 1000 / avg_time_ms

print(f"ONNX Runtime - Average Inference Time: {avg_time_ms:.4f} ms, FPS: {fps:.4f}")
