from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityRangeD,
    EnsureTypeD, LambdaD, RandSpatialCropSamplesD
)
from .uavid_labels import rgb_to_cls
from .albu_wrapper import AlbumentationsD
from .albu_aug_helper import get_weather_transforms, get_val_transforms

def train_transforms(patch_size=512, rain=False, sunny=False, snow=False, foggy=False, clahe=False):
    keys = ['image', 'label']
    mode = ('bilinear', 'nearest')

    albu_weather = get_weather_transforms(
        rain=rain,
        sunny=sunny,
        snow=snow,
        foggy=foggy,
        clahe=clahe,
    )

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        RandSpatialCropSamplesD(
            keys=keys,
            roi_size=[patch_size, patch_size],
            num_samples=6,
            random_size=False,
        ),
        AlbumentationsD(keys=keys, aug=albu_weather),
        LambdaD(keys=['label'], func=rgb_to_cls),
        EnsureTypeD(keys=keys),
    ])

def val_transforms():
    keys = ['image', 'label']

    return Compose([
        LoadImageD(keys=keys),
        EnsureChannelFirstD(keys=keys),
        LambdaD(keys=['label'], func=rgb_to_cls),
        AlbumentationsD(keys=['image'], aug=get_val_transforms()),
        EnsureTypeD(keys=keys),
    ])