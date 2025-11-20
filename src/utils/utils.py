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


def compute_iou_sample(pred, target, num_classes=8):
    eps = 1e-7
    ious = []

    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection

        iou = (intersection + eps) / (union + eps)
        ious.append(iou.item())

    mean_iou = np.mean(ious)
    return mean_iou
    
