# Factory Guardian

**Factory Guardian** is an anomaly detection project for industrial images from the _MVTec AD_ dataset based on a lightweight Variational Autoencoder (LiteVAE).
The main flow is:

1. Train on "good" samples of an MVTec AD category.
2. Compute image-level and pixel-level thresholds.
3. Test with AUROC, classification reports, and diagnostic plots.
4. Optional inference benchmark in PyTorch or ONNX Runtime.


## ✨ Features

- Lightweight VAE with depthwise separable convolutions
- Training and testing on MVTec AD categories
- Automatic threshold selection (image-level and pixel-level)
- Metrics and reports (AUROC, classification report)
- Plots for loss, ROC curves, and qualitative results
- ONNX export and ONNX Runtime benchmark


## 🧩 Project Structure

```
factory-guardian/
├─ data/                  # MVTec AD dataset by category
├─ factory_guardian/      # Source code
├─ onnx_inference/        # ONNX export and benchmark
├─ results/               # Generated outputs
├─ run.py                 # Train/test entrypoint
├─ run_onnx.py            # ONNX entrypoint
├─ setup.py               # Dataset setup
├─ environment.yml        # Environment file (conda)
├─ requirements.txt       # Python dependencies (virtualenv)
└─ README.md              # Project overview and instructions
```


## 🛠️ Installation

Python 3.11 is recommended.

#### CONDA

```bash
git clone https://github.com/ffalcone11802/factory-guardian.git && cd factory-guardian
conda env create -f environment.yml
conda activate fg-env
```

#### VIRTUALENV

```bash
git clone https://github.com/ffalcone11802/factory-guardian.git && cd factory-guardian
virtualenv -p /usr/bin/python3.11 venv # your python location and version
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## 📦 Dataset Setup

1. Download the MVTec AD dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).
2. Run the following command to unzip the file:

```bash
python setup.py --dataset_path=<path_to_the_dataset_file>
```

After this, the data folder structure should be the following:

```
data/<category>/
├─ train/
│  └─ good/
├─ test/
│  ├─ good/
│  └─ <defect_type>/
└─ ground_truth/
   └─ <defect_type>/
```


## ⚡ Quick Start

### 1) Train + Test (PyTorch)

```bash
python run.py \
  --category=toothbrush \
  --latent_dim=128 \
  --batch_size=16 \
  --num_epochs=200 \
  --lr=2e-4 \
  --init_type=xavier \
  --save_imgs_freq=10 \
  --save_checkpoint_freq=20
```

Training outputs:
- final weights in `results/weights/<category>.pth`
- thresholds in `results/params/<category>.json`
- plots in `results/plots/`

Testing outputs:
- image-level and pixel-level AUROC
- classification reports
- ROC plots and qualitative results in `results/plots/`

### 2) Test Only (PyTorch)

Since the tuned model weights and thresholds for all the categories are already provided, you can skip training and run test only:

```bash
python run.py \
  --inference_mode=True \
  --category=toothbrush \
  --latent_dim=128 \
  --batch_size=16
```

### 3) ONNX Runtime Benchmark

```bash
python run_onnx.py \
  --category=toothbrush \
  --latent_dim=128 \
  --num_warm_up_epochs=100 \
  --num_inference_epochs=1000
```

The first run exports the model to `onnx_inference/<category>_model.onnx` if missing.


## ⚙️ Main Parameters

In the following, you can find the list of main parameters and their default values (in parentheses).
Parameters can be passed as arguments to the script using the `--<arg_name>=<arg_value>` format.

#### `run.py`

- `category` (`toothbrush`) - object category to train or test on
- `rotation_range` (`10`) - degree range for random rotations in the preprocessing (set to 0 to disable)
- `latent_dim` (`128`) - dimensionality of the latent space
- `beta` (`1.0`) - beta factor for weighting the KL divergence in the ELBO loss
- `batch_size` (`16`) - batch size
- `num_epochs` (`200`) - number of epochs
- `lr` (`2e-4`) - learning rate
- `num_workers` (`0`) - number of dataloader workers
- `init_type` (`xavier`) - `normal`, `xavier`, `kaiming` - LiteVAE initialization method
- `save_imgs_freq` (`10`) - save sample reconstructions every save_imgs_freq epochs
- `save_checkpoint_freq` (`20`) - save model checkpoints every save_checkpoint_freq epochs
- `verbose` (`False`) - enable verbose output during training
- `inference_mode` (`False`) - skip training and only run inference on the test set
- `num_warm_up_epochs` (`20`) - number of epochs to warm up the model
- `num_inference_epochs` (`1000`) - number of epochs to run inference on a dummy input

#### `run_onnx.py`

- `category` (`toothbrush`) - object category to train or test on (only for compatibility with evaluation done using the PyTorch model)
- `latent_dim` (`128`) - dimensionality of the latent space
- `num_warm_up_epochs` (`20`) - number of epochs to warm up the model
- `num_inference_epochs` (`1000`) - number of epochs to run inference on a dummy input


## 👥 The Team

**Factory Guardian** has been developed by:

- Francesco Falcone ([f.falcone3@studenti.poliba.it](mailto:f.falcone3@studenti.poliba.it))

#### Deep Learning - Project Work

_Polytechnic University of Bari_

_Academical Year 2024-2025_


## 📚 References

**MVTec AD Dataset**  
Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C.  
*The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*,
International Journal of Computer Vision (IJCV), vol. 129, pp. 1038-1059, 2021.

Dataset website: https://www.mvtec.com/company/research/datasets/mvtec-ad

Dataset license and additional notes are available in:
- `data/readme.txt`
- `data/license.txt`