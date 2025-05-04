import sys

import albumentations as A


def get_categorized_augmentations():
    set_of_all_categorized_augmentations = {
        "class1_color_adjustments": [
            A.Blur(blur_limit=7, always_apply=True, p=1),
            A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True, p=1),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=True, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True, p=1),
            A.Solarize(threshold=128, always_apply=True, p=1),
            A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=True, p=1),
            A.Posterize(num_bits=4, always_apply=True, p=1),
            A.FancyPCA(alpha=0.1, always_apply=True, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True, p=1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=1),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=True, p=1),
            A.Superpixels(p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=True, p=1),
            A.RingingOvershoot(blur_limit=(7, 15), cutoff=(0.7853981633974483, 1.5707963267948966), always_apply=True, p=1),
            A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=True, p=1),
            A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=True, p=1),
            A.RandomToneCurve(scale=0.1, always_apply=True, p=1),
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=True, p=1)
        ],
        "class2_geometric_transformations": [
            A.VerticalFlip(always_apply=True, p=1),
            A.HorizontalFlip(always_apply=True, p=1),
            A.Flip(always_apply=True, p=1),
            A.Transpose(always_apply=True, p=1),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, rotate_method='largest_box', always_apply=True, p=1),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=True, approximate=False, same_dxdy=False, p=1),
        ],

        "class3_noise_distortion":[
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=True, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=True, p=1),
            A.RandomGridShuffle(grid=(3, 3), always_apply=True, p=1),
            A.MotionBlur(blur_limit=7, always_apply=True, p=1),
            A.MedianBlur(blur_limit=7, always_apply=True, p=1),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=True, p=1),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=True, p=1),
            A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=True, mode='fast', p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1),
            A.ChannelShuffle(p=1, always_apply=True),
            A.InvertImg(p=1, always_apply=True),
            A.ToSepia(always_apply=True, p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=True, p=1),
            A.AdvancedBlur(blur_limit=(3, 9), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=90, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.2), always_apply=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.5, 1.1), per_channel=False, elementwise=False, always_apply=True, p=1),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=True, p=1),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=True, p=1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=True, p=1),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=True, p=1),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=True, p=1),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=True, p=1),
            A.GridDropout(ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=True, p=1),
            # A.MaskDropout(max_objects=1, image_fill_value=0, mask_fill_value=0, always_apply=True, p=1),
        ],
        "class4_affine_transformations": [
            A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=True, p=1),
            A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=False, always_apply=True, p=1),
            A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=True, keypoints_threshold=0.01, p=1),
        ],
        # 1. We set cropping size at 80% of the original image height and width
        # 2. We also preset the x_max and y_max to 80% of the required input image
        # 3. RandomSizedCrop ?
        "class5_cropping_transformations": [
            A.CenterCrop(height=50, width=50, always_apply=True, p=1.0),
            A.Crop(x_min=0, y_min=0, x_max=50, y_max=50, always_apply=True, p=1.0),
            # A.CropAndPad(px=None, percent=10, pad_mode=0, pad_cval=0, pad_cval_mask=0, keep_size=False, sample_independently=True, interpolation=1, always_apply=True, p=1.0),
            A.RandomCrop(height=50, width=50, always_apply=True, p=1.0),
            # A.RandomCropNearBBox(max_part_shift=(0.3, 0.3), cropping_box_key='cropping_bbox', always_apply=True, p=1.0),
            A.RandomResizedCrop(height=50, width=50, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=True, p=1.0),
            # A.RandomSizedCrop(min_max_height=min_max_height, height=img_height, width=img_width, w2h_ratio=1.0, interpolation=1, always_apply=True, p=1.0),
        ],
        #  PadIfNeeded ?
        "class6_resizing_transformations": [
            A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=2, always_apply=True, p=1),
            # A.LongestMaxSize(max_size=max_size, interpolation=1, always_apply=True, p=1),
            A.RandomScale(scale_limit=0.4   , interpolation=1, always_apply=True, p=1),
            # A.SmallestMaxSize(max_size=max_size, interpolation=1, always_apply=True, p=1),
            # A.PadIfNeeded(min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None, position='center', border_mode=4, value=None, mask_value=None, always_apply=True, p=1.0),
        ],
        "class7_rotation_transformations": [
            A.RandomRotate90(always_apply=True, p=1),
            A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, crop_border=False, always_apply=True, p=1),
            A.SafeRotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=True, p=1),
        ]
    }

    return set_of_all_categorized_augmentations



