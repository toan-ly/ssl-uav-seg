import numpy as np
from monai.transforms import MapTransform   

class AlbumentationsD(MapTransform):
    def __init__(self, keys, aug):
        super().__init__(keys)
        self.aug = aug
    
    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        img = np.asarray(img)
        img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C

        if len(self.keys) == 1:
            augmented = self.aug(image=img)
            img_aug = augmented['image']
            img_aug = np.transpose(img_aug, (2, 0, 1))  # [C,H,W]
            d[self.keys[0]] = img_aug.astype(np.float32)
            return d

        mask = d[self.keys[1]]
        mask = np.asarray(mask)

        mask = np.transpose(mask, (1, 2, 0))  # C,H,W -> H,W,C

        augmented = self.aug(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        img_aug = np.transpose(img_aug, (2, 0, 1))  # [C,H,W]
        mask_aug = np.transpose(mask_aug, (2, 0, 1))  # [C,H,W] RGB mask

        d[self.keys[0]] = img_aug.astype(np.float32)   # image
        d[self.keys[1]] = mask_aug.astype(np.uint8)    # label (RGB)

        return d