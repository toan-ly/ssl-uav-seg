from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityRangeD,
    EnsureTypeD, LambdaD, RandSpatialCropSamplesD
)
from .uavid_labels import rgb_to_cls
from .albu_wrapper import AlbumentationsD # get albu wrapper
from .albu_aug_helper import get_weather_transforms, get_val_transforms
from .copy_paste import CopyPaste
import random

def _get_cp_transform(data_pairs):
    return CopyPaste(
        data=random.sample(data_pairs, 50),
        paste_ratio=[0, 0, 0, 0, 0, 0.2, 0.2, 0.3],
        num_paste=20,
        instance_per_cls=5,
        pool_samples=10,
        always_apply=False,
        p=0.5,
    )
    
def train_transforms(
    patch_size=512, 
    rain=False, 
    sunny=False, 
    snow=False, 
    foggy=False, 
    clahe=False, 
    copy_paste=False,
    data_pairs=None,
):

    keys = ['image', 'label']

    cp_transform = None
    if copy_paste:
        cp_transform = _get_cp_transform(data_pairs=data_pairs)

    albu_transform = get_weather_transforms(
        rain=rain,
        sunny=sunny,
        snow=snow,
        foggy=foggy,
        clahe=clahe,
        copy_paste=cp_transform,
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
        AlbumentationsD(keys=keys, aug=albu_transform),
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