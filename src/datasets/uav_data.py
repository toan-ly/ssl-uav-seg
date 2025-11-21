from pathlib import Path
import os
from glob import glob

import torch

from monai.data import DataLoader, CacheDataset, Dataset
from .aug import train_transforms, val_transforms


def get_pairs(dir):
    im_paths = sorted(glob(str(dir / 'seq*' / 'Images' / '*.png')))

    items = []
    for im_path in im_paths:
        mask_path = im_path.replace('Images', 'Labels')
        if not os.path.exists(mask_path) and 'test' not in str(dir):
            raise FileNotFoundError(f'Mask not found for image: {im_path}')
        items.append({'image': im_path, 'label': mask_path})

    print(f'{dir}: {len(items)} samples')
    return items

def make_loaders(
    data_root,
    batch_size=4,
    patch_size=512,
    num_workers=2,
    cache_rate=0.0,
    rain=False,
    sunny=False,
    snow=False,
    foggy=False,
    clahe=False,
    copy_paste=False,
):
    train_dir = Path(data_root) / 'uavid_train'
    val_dir = Path(data_root) / 'uavid_val'

    train_items = get_pairs(train_dir)
    val_items = get_pairs(val_dir)

    train_t = train_transforms(
        patch_size=patch_size,
        rain=rain,
        sunny=sunny,
        snow=snow,
        foggy=foggy,
        clahe=clahe,
        copy_paste=copy_paste,
        data_pairs=train_items,
    )
    val_t = val_transforms()

    train_ds = Dataset(data=train_items, transform=train_t)
    if cache_rate > 0:
        val_ds = CacheDataset(
            data=val_items,
            transform=val_t,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        val_ds = Dataset(data=val_items, transform=val_t)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader

if __name__ == '__main__':
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'UAVid'
    train_loader, val_loader = make_loaders(DATA_DIR, cache_rate=0.0)
    for batch in train_loader:
        print(batch['image'].shape, batch['label'].shape)
        unique_cls = torch.unique(batch['label'])
        print('Unique classes in labels:', unique_cls)
        break
    for batch in val_loader:
        print(batch['image'].shape, batch['label'].shape)
        unique_cls = torch.unique(batch['label'])
        print('Unique classes in labels:', unique_cls)
        break

