import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from .models import *
from pathlib import Path
from .datasets.uav_data import make_loaders
from .utils.trainer import Trainer

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'UAVid'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8
IN_CHANNELS = 3

model = create_unet(
    num_classes=NUM_CLASSES,
    in_channels=IN_CHANNELS,
    encoder_name="resnet50",
    encoder_weights="imagenet",
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {n_params}')

train_loader, val_loader = make_loaders(
    DATA_DIR, 
    cache_rate=0.0,
    batch_size=4,
    patch_size=512,
    num_workers=0,
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    loss='dicece',
    optimizer_name='adamw',
    lr=1e-3,
    early_stopping=True,
    patience=5,
    scheduler='cosine',
    cls_weights=None,
    num_classes=NUM_CLASSES,
)

trainer.fit(
    epochs=50, 
    verbose=True, 
    save_model_path="unet_resnet50_base.pth", 
    save_plots_path="figures"
)