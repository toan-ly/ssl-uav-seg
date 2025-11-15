import numpy as np

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

COLOR2ID = {rgb: idx for idx, rgb in enumerate(UAVID_COLORMAP)}
ID2COLOR = {idx: rgb for idx, rgb in enumerate(UAVID_COLORMAP)}

def rgb_to_id(mask):
    h, w, _ = mask.shape
    mapped_mask = np.zeros((h, w), dtype=np.uint8)

    for rgb, cls_id in COLOR2ID.items():
        matches = np.all(mask == rgb, axis=-1)
        mapped_mask[matches] = cls_id

    return mapped_mask