from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityRangeD,
    RandCropByPosNegLabelD, RandFlipD, RandAffineD, EnsureTypeD, LambdaD,
    RandSpatialCropSamplesD
)
from .uavid_labels import rgb_to_cls

def train_transforms(patch_size=512):
    keys = ['image', 'label']
    mode = ('bilinear', 'nearest')

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        LambdaD(keys=['label'], func=rgb_to_cls),

        ScaleIntensityRangeD(
            keys=['image'],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        
        RandSpatialCropSamplesD(
            keys=keys,
            roi_size=[patch_size, patch_size],
            num_samples=8,
            random_size=False,
        ),

        RandFlipD(keys=keys, prob=0.5, spatial_axis=0), # horizontal flip

     

        EnsureTypeD(keys=keys),
    ])

def val_transforms():
    keys = ['image', 'label']

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        LambdaD(keys=['label'], func=rgb_to_cls),
        ScaleIntensityRangeD(
            keys=['image'],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        EnsureTypeD(keys=keys),
    ])