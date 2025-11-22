import torch
import segmentation_models_pytorch as smp

def create_unet(
    num_classes: int = 8, 
    in_channels: int = 3,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
