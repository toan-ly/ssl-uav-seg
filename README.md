# 🛰️ Self-Supervised Learning for UAV Semantic Segmentation
This repository contains the code, models, and experimental results for our project exploring MoCo v2 self-supervised pretraining to improve semantic segmentation of unmanned aerial vehicle (UAV) imagery. We finetune multiple architectures (UNet, UNet++, DeepLabV3) on the UAVid dataset and evaluate how self-supervised features compare to ImageNet and random initialization.

## 🌟 Key Contributions
- **Self-supervised encoder pretraining** using *MoCo v2* on a large mixed UAV imagery corpus (UAVid + Semantic Drone + UDD)
- Finetuning 3 segmentation models (UNet, UNet++, DeepLabV3) with SSL initialized encoders
- Consistent improvements in mIoU, F1, and OA across all architectures

## 🗃️ Datasets
### Unlabeled Data (SSL Pretraining)
Used for contrastive learning in MoCo v2:
- UAVid (train + val)
- Semantic Drone Dataset
- UDD (Urban Drone Dataset)

Total unlabeled images: **~960**

### Labeled Data (Supervised Finetuning)
All segmentation training and evaluation use **UAVid**:
- Train: 200 images
- Validation: 70 images
- Test: 150 images
- Classes (8): Clutter, Building, Road, Tree, Low Vegetation, Moving Car, Static Car, Human
- Dense, high-resolution 4K imagery from urban UAV passes


## 🛠️ Environment Setup

We're using `uv` (recommended)

For MacOS/Linux, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows, run:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

The above command will install `uv` in your machine, or you can go to [this link](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) for more detailed

After installing `uv`, run this command to install all the packages:
```bash
uv sync
```

Or if you prefer other env packages like Conda or Venv, below is an example of venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Run the Code

This repository contains two major components:

1.	Self-Supervised Pretraining (MoCo v2)
2.	Supervised Semantic Segmentation (UNet, UNet++, DeepLabV3)

## 1️⃣ Self-Supervised Pretraining (MoCo v2)
### Prepare Unlabeled Dataset List

Your data folder must follow this structure:

```
data/
│── UAVid/
│   ├── uavid_train/
│   ├── uavid_val/
│   ├── uavid_test/
│   └── ...
│
│── semantic_drone/
│   └── images/
│
│── UDD/
│   └── *.JPG
```

### Run MoCo v2 Training
```bash
uv run -m src.train_moco
```

## 2️⃣ Supervised Training (UNet, UNet++, DeepLabV3)
You can train models using:
- ImageNet Pretrained
- SSL Pretrained (your MoCo weights)
- Random Initialization

### (A) Train UNet/UNet++
You will need to modify the code in `src/train_unet_*.py` for specific model and run one of these:
```bash
uv run -m src.train_unet_imagenet
uv run -m src.train_unet_random
uv run -m src.train_unet_ssl
```

### (B) Train DeepLabV3
Similarly,
```bash
uv run -m src.train.train_deeplab_imagenet
uv run -m src.train.train_deeplab_ssl
```

## 3️⃣ Evaluating a Trained Model
Run below code to perform inference:
```bash
uv run -m src.test.evaluate
uv run -m src.test.summarize
```