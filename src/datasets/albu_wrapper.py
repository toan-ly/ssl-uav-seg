import numpy as np
from monai.transforms import MapTransform   

class AlbumentationsD(MapTransform):
    def __init__(self, keys, aug):
        super().__init__(keys)
        self.aug = aug
    
    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        mask = d[self.keys[1]]

        img = np.asarray(img)
        mask = np.asarray(mask)

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C

        if mask.ndim == 3 and mask.shape[0] <= 3 and mask.shape[0] < mask.shape[1]:
            mask = np.transpose(mask, (1, 2, 0))  # C,H,W -> H,W,C

        augmented = self.aug(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        if img_aug.ndim == 3:
            img_aug = np.transpose(img_aug, (2, 0, 1))  # [C,H,W]

        if mask_aug.ndim == 3:
            mask_aug = np.transpose(mask_aug, (2, 0, 1))  # [C,H,W] RGB mask

        d[self.keys[0]] = img_aug.astype(np.float32)   # image
        d[self.keys[1]] = mask_aug.astype(np.uint8)    # label (RGB)

        return d