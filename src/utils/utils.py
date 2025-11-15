import torch
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)

def get_device(device_str=None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_dice_iou(preds, targets):
    """
    Compute Dice coefficient and Intersection over Union (IoU) between predictions and targets.

    Args:
        preds: Predicted binary masks (B, C, H, W)
        targets: Ground truth binary masks (B, C, H, W)
    """
    eps = 1e-7
    intersection = (preds * targets).sum((1, 2, 3)) # Sum over C, H, W
    sum_preds_targets = preds.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    dice = (2. * intersection) / (sum_preds_targets + eps)
    iou = intersection / (sum_preds_targets - intersection + eps)
    return dice.mean().item(), iou.mean().item()

def compute_dice_iou_sample(pred, target):
    """
    Compute Dice coefficient and Intersection over Union (IoU) for a single sample.

    Args:
        pred: Predicted binary mask (C, H, W)
        target: Ground truth binary mask (C, H, W)
    """
    eps = 1e-7
    intersection = (pred * target).sum()
    sum_preds_targets = pred.sum() + target.sum()
    dice = (2. * intersection) / (sum_preds_targets + eps)
    iou = intersection / (sum_preds_targets - intersection + eps)
    return dice.item(), iou.item()

