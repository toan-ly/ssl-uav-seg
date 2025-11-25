import albumentations as A

def get_weather_transforms(
    rain=False,
    sunny=False,
    snow=False,
    foggy=False,
    clahe=False,
    copy_paste=None,
):
    Aug = []

    if copy_paste:
        Aug.append(copy_paste)

    if clahe:
        Aug.append(
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.5)
        )

    Aug.extend([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.0),
        A.GaussianBlur(blur_limit=(3,5), p=0.0),
        A.VerticalFlip(p=0.0),
        A.RandomRotate90(p=0.0),
    ])

    W = []
    if rain:
        W.append(
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=15,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=12,
                brightness_coefficient=0.52,
                rain_type='heavy',
                p=0.25,
            )
        )
    
    if sunny:
        W.append(
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                angle_range=(0, 1),
                num_flare_circles_range=(4, 8),
                src_radius=100,
                src_color=(255, 255, 255),
                p=0.25,
            )
        )

    if snow:
        W.append(
            A.RandomSnow(
                snow_point_range=(0.1, 0.3),
                brightness_coeff=2.5,
                p=0.25,
            )
        )

    if foggy:
        W.append(
            A.RandomFog(
                fog_coef_range=(0.3, 0.4),
                alpha_coef=0.1,
                p=0.01,
            )
        )

    Aug.extend(W)
    Aug.append(A.Normalize(p=1.0))
    return A.Compose(Aug)

def get_val_transforms():
    return A.Compose([
        A.Normalize(p=1.0),
    ])