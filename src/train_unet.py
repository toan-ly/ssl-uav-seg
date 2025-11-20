import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from .models import *
from pathlib import Path
from .datasets.uav_data import make_loaders
from .utils.trainer import Trainer
from .utils.utils import set_seed

set_seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'UAVid'

Path('weights').mkdir(parents=True, exist_ok=True)
Path('figures').mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {DEVICE}')
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
    cache_rate=1,
    batch_size=8,
    patch_size=512,
    num_workers=4,
    clahe=True,
)

weights_cls = [2, 1, 1, 1, 1, 3, 3, 5]

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,  
    loss='dicece',
    optimizer_name='adamw',
    lr=4e-4,
    early_stopping=True,
    patience=5,
    scheduler='cosine',
    cls_weights=weights_cls,
    num_classes=NUM_CLASSES,
)


start_epoch = 0
checkpoint_path = "weights/unet_resnet50_base_last.pth"
if Path(checkpoint_path).exists():
    start_epoch = trainer.load_checkpoint(checkpoint_path)
else:
    print('No checkpoint found, training from scratch.')

trainer.fit(
    epochs=100, 
    verbose=True, 
    save_model_path="weights/unet_resnet50_base.pth", 
    save_plots_path="figures/unet_base"
)