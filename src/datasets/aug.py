from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, RandAffined, EnsureTyped, Lambdad,
)
from .uavid_labels import rgb_to_id

def train_transforms(patch_size=512):
    keys = ['image', 'label']
    mode = ('bilinear', 'nearest')

    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys=['label'], func=lambda x: rgb_to_id(x)),

        ScaleIntensityRanged(
            keys=['image'],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        RandFlipd(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip

        RandCropByPosNegLabeld(
            keys=keys,
            label_key='label',
            spatial_size=(patch_size, patch_size),
            pos=1,
            neg=1,
            num_samples=4,
        ),

        EnsureTyped(keys=keys),
    ])

def val_transforms():
    keys = ['image', 'label']

    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys=['label'], func=lambda x: rgb_to_id(x)),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=keys),
    ])