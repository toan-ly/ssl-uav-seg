import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from PIL import Image  
import matplotlib.pyplot as plt
import pandas as pd

from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference


from .models import create_unet
from .utils.metric import Evaluator
from .utils.utils import set_seed
from .datasets.uav_data import get_pairs
from .datasets.aug import val_transforms

from glob import glob

UAVID_CLS = [
    'Background clutter', # 0
    'Building',           # 1
    'Road',               # 2
    'Tree',               # 3
    'Low vegetation',     # 4
    'Moving car',         # 5
    'Static car',         # 6
    'Human',              # 7
]

UAVID_COLORMAP = [
    (0, 0, 0),          # Background Clutter
    (128, 0, 0),        # Building
    (128, 64, 128),     # Road
    (0, 128, 0),        # Tree
    (128, 128, 0),      # Low vegetation
    (64, 0, 128),       # Moving car
    (192, 0, 192),      # Static car
    (64, 64, 0),        # Human 
]

def decode_segmap(pred_mask):
    h, w = pred_mask.shape
    mapped_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, rgb in enumerate(UAVID_COLORMAP):
        matches = (pred_mask == cls_id)
        mapped_mask[matches] = rgb
    
    return mapped_mask

def build_test_loader(
    data_root, 
    batch_size: int = 1,
    num_workers: int = 4,
    cache_rate: float = 0.0,
):
    test_dir = Path(data_root) / 'test_label'
    test_pairs = get_pairs(test_dir)
    test_t = val_transforms()

    if cache_rate > 0:
        test_dataset = CacheDataset(
            data=test_pairs,
            transform=test_t,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        test_dataset = Dataset(data=test_pairs, transform=test_t)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader, test_pairs

def load_model(model, weight_path, device):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f'Weight file not found: {weight_path}')

    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    print(f'Loaded model weights from {weight_path}')
    return model

def plot_results(im, gt_rgb, pred_rgb, im_name, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(im)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(gt_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path / f'{im_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def evaluate(
    model_name: str | Path,
    ckpt_path: str | Path,
    data_root: str | Path,
    results_dir: str | Path,
    device: str = 'cuda',
    num_classes: int = 8,
    test_loader=None,
    test_pairs=None,
):
    device = device if torch.cuda.is_available() else 'cpu'
    print(f'\n=== Evaluating Model: {model_name} ===')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Using device: {device}')

    pred_dir = Path(results_dir) / model_name / 'predictions'
    if pred_dir.exists():
        print(f'[Skipping] Predictions already exist at {pred_dir}')
        return

    output_dir = Path(results_dir) / model_name / 'figures'
    pred_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_loader is None or test_pairs is None:
        test_loader, test_pairs = build_test_loader(
            data_root=data_root,
            batch_size=1,
            num_workers=4,
        )

    model = create_unet(
        num_classes=num_classes,
        in_channels=3,
        encoder_name="resnet50",
        encoder_weights=None,
    ).to(device)

    model = load_model(model, ckpt_path, device)
    model.eval()

    evaluator = Evaluator(num_class=num_classes)
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc='Evaluating', leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Sliding window inference on full test image
            logits = sliding_window_inference(
                images,
                roi_size=(1024, 1024),
                sw_batch_size=24,
                predictor=model,
                overlap=0.125,
            )

            preds = torch.argmax(logits, dim=1, keepdim=False) # [B, H, W]
            gt = labels.squeeze(1) # [B, H, W]

            for i in range(gt.shape[0]):
                if global_idx >= len(test_pairs):
                    break

                # Compute metrics
                evaluator.add_batch(
                    gt[i].cpu().numpy(), 
                    preds[i].cpu().numpy()
                )

                pair = test_pairs[global_idx]
                im_path = Path(pair['image'])
                seq_name = im_path.parts[-3]
                filename = im_path.stem
                img_name = f'{seq_name}_{filename}'

                im_np = images[i].detach().cpu().numpy().transpose(1, 2, 0) # [H, W, C]

                im_min, im_max = im_np.min(), im_np.max()
                if im_max > im_min:
                    im_np = (im_np - im_min) / (im_max - im_min)

                gt_np = gt[i].cpu().numpy().astype(np.uint8) # [H, W]
                preds_np = preds[i].cpu().numpy().astype(np.uint8) # [H, W]

                pred_rgb = decode_segmap(preds_np) # [H, W, 3]
                gt_rgb = decode_segmap(gt_np)      # [H, W, 3]

                im_np = np.rot90(im_np, k=-1)
                gt_rgb = np.rot90(gt_rgb, k=-1)
                pred_rgb = np.rot90(pred_rgb, k=-1)

                # Save prediction mask
                plt.imsave(pred_dir / f'{img_name}.png', pred_rgb)

                # Save visualization figure
                plot_results(im_np, gt_rgb, pred_rgb, img_name, output_dir)

                global_idx += 1

            # break

        
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()

        miou = np.nanmean(iou_per_class)
        mean_f1 = np.nanmean(f1_per_class)
        oa = np.nanmean(evaluator.OA())

        print(f"\n[{model_name}] Test mIoU: {miou:.4f}, mean F1: {mean_f1:.4f}, mean OA: {oa:.4f}")

        # Build metrics DataFrame
        rows = []
        for cls_id in range(num_classes):
            rows.append(
                {
                    'class_id': cls_id,
                    'class_name': UAVID_CLS[cls_id],
                    'IoU': iou_per_class[cls_id],
                    'F1': f1_per_class[cls_id],
                    'OA': np.nan,
                }
            )

        rows.append(
            {
                'class_id': -1,
                'class_name': 'mean',
                'IoU': miou,
                'F1': mean_f1,
                'OA': oa,
            }
        )

        df_metrics = pd.DataFrame(rows)
        metrics_csv_path = Path(results_dir) / model_name / 'test_metrics.csv'
        df_metrics.to_csv(metrics_csv_path, index=False)
        print(f'Saved test metrics to {metrics_csv_path}')


def main():
    set_seed(42)

    ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = ROOT / 'data' / 'UAVid'
    RESULTS_DIR = ROOT / 'results'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR = ROOT / 'weights'

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 8

    MODEL_NAMES = [Path(p).name for p in glob(str(CKPT_DIR / 'unet_*'))]
    MODELS = {model_name: CKPT_DIR / model_name / f'{model_name}_best.pth' for model_name in MODEL_NAMES}
    test_loader, test_pairs = build_test_loader(
        data_root=DATA_ROOT,
        batch_size=1,
        num_workers=4,
        cache_rate=1,
    )

    for model_name, ckpt_path in MODELS.items():
        if not ckpt_path.exists():
            print(f'[ERROR] Checkpoint not found for {model_name}: {ckpt_path}')
            continue
        
        evaluate(
            model_name=model_name,
            ckpt_path=ckpt_path,
            data_root=DATA_ROOT,
            results_dir=RESULTS_DIR,
            device=DEVICE,
            num_classes=num_classes,
            test_loader=test_loader,
            test_pairs=test_pairs,
        )


if __name__ == '__main__':
    main()