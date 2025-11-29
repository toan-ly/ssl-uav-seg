import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from pathlib import Path

from .datasets.uav_data import make_loaders
from .utils.trainer import Trainer
from .utils.utils import set_seed

set_seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'UAVid'

Path('weights').mkdir(parents=True, exist_ok=True)
Path('figures').mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {DEVICE}')
NUM_CLASSES = 8
IN_CHANNELS = 3

model = smp.DeepLabV3(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
).to(DEVICE)

ssl_weight_path = ROOT / 'weights' / 'ssl_mocov2' / 'moco_v2_resnet50_55_best.pth'
if not ssl_weight_path.exists():
    raise FileNotFoundError(f'SSL weights not found at {ssl_weight_path}')
checkpoint = torch.load(ssl_weight_path, map_location=DEVICE)
state_dict = checkpoint['encoder_q']

encoder_state = {}
for k, v in state_dict.items():
    if k.startswith("module.encoder_q."):
        new_k = k[len("module.encoder_q."):]
    elif k.startswith("encoder_q."):
        new_k = k[len("encoder_q."):]
    else:
        new_k = k
    encoder_state[new_k] = v

print(f'State dict keys to be loaded into encoder: {list(encoder_state.keys())[:10]} ...')

load_res = model.encoder.load_state_dict(encoder_state, strict=False)
print(f"Loaded SSL weights from {ssl_weight_path}")
print(f'    missing keys: {getattr(load_res, "missing_keys", [])}')
print(f'    unexpected keys: {getattr(load_res, "unexpected_keys", [])}')

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {n_params}')

train_loader, val_loader = make_loaders(
    DATA_DIR, 
    cache_rate=1,
    batch_size=6,
    patch_size=512,
    num_workers=4,
)

weights_cls = [2, 1, 1, 1, 1, 3, 3, 5]

for p in model.encoder.parameters():
    p.requires_grad = False
print('Phase 1: Frozen encoder training.')
print('Number of trainable parameters after freezing encoder:', 
      sum(p.numel() for p in model.parameters() if p.requires_grad))

trainer_1 = Trainer(
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

print('Starting Phase 1 training: Frozen encoder.')
start_epoch = 0
checkpoint_path = "weights/deeplabv3_resnet50_ssl_phase1.pth"
if Path(checkpoint_path).exists():
    start_epoch = trainer_1.load_checkpoint(checkpoint_path)
    print(f'Resuming Phase 1 training from epoch {start_epoch}...')
else:
    print('No checkpoint found, training from scratch.')

trainer_1.fit(
    epochs=10,
    verbose=True,
    save_model_path=checkpoint_path,
    save_plots_path="figures/deeplabv3_resnet50_ssl_phase1",
)

phase1_best = "weights/deeplabv3_resnet50_ssl_phase1_best.pth"
if Path(phase1_best).exists():
    trainer_1.load_checkpoint(phase1_best)
    model = trainer_1.model
    print(f'Loaded best model from Phase 1: {phase1_best}')

# ----------------------------------
# PHASE 2: Unfreeze encoder and fine-tune entire model
# ----------------------------------
for p in model.encoder.parameters():
    p.requires_grad = True
print('Phase 2: Unfrozen encoder training.')
print('Number of trainable parameters after unfreezing encoder:', 
      sum(p.numel() for p in model.parameters() if p.requires_grad))    

trainer_2 = Trainer(
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
    start_epoch = trainer_2.load_checkpoint(phase1_best)
    print(f'Resuming from checkpoint: {phase1_best}')
else:
    print('No checkpoint found, starting Phase 2 training from scratch.')

trainer_2.fit(
    epochs=100,
    verbose=True,
    save_model_path="weights/deeplab_ssl.pth",
    save_plots_path="figures/deeplab_ssl",
)