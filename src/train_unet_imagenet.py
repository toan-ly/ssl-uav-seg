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
    encoder_weights="imagenet", # imagenet pretrain
).to(DEVICE)


n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {n_params}')

train_loader, val_loader = make_loaders(
    DATA_DIR, 
    cache_rate=1,
    batch_size=8,
    patch_size=512,
    num_workers=4,
)

weights_cls = [2, 1, 1, 1, 1, 3, 3, 5]

# ----------------------------------
# PHASE 1: Freeze encoder and train decoder
# ----------------------------------
for param in model.encoder.parameters():
    param.requires_grad = False
print('Phase 1: Frozen encoder training.')
print('Number of trainable parameters after freezing encoder:', 
      sum(p.numel() for p in model.parameters() if p.requires_grad))


trainer1 = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,  
    loss='dicece',
    optimizer_name='adamw',
    lr=4e-4,
    early_stopping=True,
    patience=2, 
    scheduler='cosine',
    cls_weights=weights_cls,
    num_classes=NUM_CLASSES,
    encoder_lr=None,
)

print('Starting Phase 1 training...')
start_epoch = 0
checkpoint_path = "weights/unet_resnet50_base_last.pth"
if Path(checkpoint_path).exists():
    start_epoch = trainer1.load_checkpoint(checkpoint_path)
    print(f'Resuming Phase 1 training from epoch {start_epoch}...')
else:
    print('No checkpoint found, training from scratch.')

trainer1.fit(
    epochs=10, 
    verbose=True, 
    save_model_path="weights/unet_resnet50_base_phase1.pth", 
    save_plots_path="figures/unet_base"
)

phase1_best = "weights/unet_resnet50_base_phase1_best.pth"
if Path(phase1_best).exists():
    trainer1.load_checkpoint(phase1_best)
    model = trainer1.model
    print(f'Loaded best model from Phase 1: {phase1_best}')

# ----------------------------------
# PHASE 2: Unfreeze encoder and fine-tune entire model
# ----------------------------------
for p in model.encoder.parameters():
    p.requires_grad = True

print('Phase 2: Unfrozen encoder training.')
print('Number of trainable parameters after unfreezing encoder:', 
      sum(p.numel() for p in model.parameters() if p.requires_grad))

trainer2 = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,  
    loss='dicece',
    optimizer_name='adamw',
    lr=4e-4,
    early_stopping=True,
    patience=7,
    scheduler='cosine',
    cls_weights=weights_cls,
    num_classes=NUM_CLASSES,
    encoder_lr=1e-4,
)

print('Starting Phase 2 Training...')
start_epoch = 0
if Path(phase1_best).exists():
    start_epoch = trainer2.load_checkpoint(phase1_best)
    print(f'Resuming from checkpoint: {phase1_best}')
else:
    print('No checkpoint found for Phase 2, training from scratch.')

trainer2.fit(
    epochs=100, 
    verbose=True, 
    save_model_path="weights/unet_resnet50_base.pth", 
    save_plots_path="figures/unet_base"
)