import numpy as np
from copy import deepcopy

from .data_augmentations.custom_transforms import (
    Convert3DTo2DTransform,
    Convert2DTo3DTransform,
    MakePseudoRGBInput,
    Make25dInput,
)
from .data_augmentations.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from .data_augmentations.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    BrightnessTransform,
    GammaTransform,
)
from .data_augmentations.resample_transforms import SimulateLowResolutionTransform
from .data_augmentations.spatial_transforms import SpatialTransform, MirrorTransform
from .data_augmentations.utility_transforms import NumpyToTensor

default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": True,
    "elastic_deform_alpha": (0.0, 900.0),
    "elastic_deform_sigma": (9.0, 13.0),
    "p_eldef": 0.2,
    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,
    "do_rotation": True,
    "rotation_x": (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
    "rotation_y": (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
    "rotation_z": (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (0, 1, 2),
    "dummy_2D": False,
    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0.0, 200.0)
default_2D_augmentation_params["elastic_deform_sigma"] = (9.0, 13.0)
default_2D_augmentation_params["rotation_x"] = (
    -180.0 / 360 * 2.0 * np.pi,
    180.0 / 360 * 2.0 * np.pi,
)
default_2D_augmentation_params["rotation_y"] = (
    -0.0 / 360 * 2.0 * np.pi,
    0.0 / 360 * 2.0 * np.pi,
)
default_2D_augmentation_params["rotation_z"] = (
    -0.0 / 360 * 2.0 * np.pi,
    0.0 / 360 * 2.0 * np.pi,
)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (
    0,
    1,
)  # this can be (0, 1, 2) if dummy_2D=True


def get_rtnet_transforms(
    params,
    final_patch_size,
    patch_center_dist_from_border,
    order_data,
    border_val_mask,
    order_mask,
):
    tr_transforms = []

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = final_patch_size[1:]
    else:
        patch_size_spatial = final_patch_size
        ignore_axes = None

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=patch_center_dist_from_border,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=order_data,
            border_mode_mask="constant",
            border_cval_mask=border_val_mask,
            order_mask=order_mask,
            random_crop=params.get("random_crop"),
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis"
            ),
        )
    )

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        GaussianBlurTransform(
            (0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5,
        )
    )
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.75, 1.25), p_per_sample=0.15
        )
    )

    if params.get("do_additive_brightness"):
        tr_transforms.append(
            BrightnessTransform(
                params.get("additive_brightness_mu"),
                params.get("additive_brightness_sigma"),
                True,
                p_per_sample=params.get("additive_brightness_p_per_sample"),
                p_per_channel=params.get("additive_brightness_p_per_channel"),
            )
        )

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=ignore_axes,
        )
    )
    tr_transforms.append(
        GammaTransform(
            params.get("gamma_range"),
            True,
            True,
            retain_stats=params.get("gamma_retain_stats"),
            p_per_sample=0.1,
        )
    )  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"],
            )
        )

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if len(patch_size_spatial) == 2:
        tr_transforms.append(MakePseudoRGBInput())

    tr_transforms.append(NumpyToTensor(["data", "mask"], "float"))

    return tr_transforms


def get_rtnet_25d_transforms(
    params,
    final_patch_size,
    patch_center_dist_from_border,
    order_data,
    border_val_mask,
    order_mask,
):
    tr_transforms = []

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = final_patch_size[1:]
    else:
        patch_size_spatial = final_patch_size
        ignore_axes = None

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=patch_center_dist_from_border,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=order_data,
            border_mode_mask="constant",
            border_cval_mask=border_val_mask,
            order_mask=order_mask,
            random_crop=params.get("random_crop"),
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis"
            ),
        )
    )

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        GaussianBlurTransform(
            (0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5,
        )
    )
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.75, 1.25), p_per_sample=0.15
        )
    )

    if params.get("do_additive_brightness"):
        tr_transforms.append(
            BrightnessTransform(
                params.get("additive_brightness_mu"),
                params.get("additive_brightness_sigma"),
                True,
                p_per_sample=params.get("additive_brightness_p_per_sample"),
                p_per_channel=params.get("additive_brightness_p_per_channel"),
            )
        )

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=ignore_axes,
        )
    )
    tr_transforms.append(
        GammaTransform(
            params.get("gamma_range"),
            True,
            True,
            retain_stats=params.get("gamma_retain_stats"),
            p_per_sample=0.1,
        )
    )  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"],
            )
        )

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    tr_transforms.append(
        Make25dInput(
            slice_per_25d_group=params.get("slice_per_25d_group"),
            num_25d_group=params.get("num_25d_group"),
            group_25d_overlap=params.get("group_25d_overlap"),
        )
    )

    tr_transforms.append(NumpyToTensor(["data", "mask"], "float"))

    return tr_transforms


def get_rtnet_25d_size_invariant_transforms(
    params,
    final_patch_size,
    patch_center_dist_from_border,
    order_data,
    border_val_mask,
    order_mask,
):
    tr_transforms = []

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = final_patch_size[1:]
    else:
        patch_size_spatial = final_patch_size
        ignore_axes = None

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=patch_center_dist_from_border,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=order_data,
            border_mode_mask="constant",
            border_cval_mask=border_val_mask,
            order_mask=order_mask,
            random_crop=params.get("random_crop"),
            size_invariant_crop=True,
            size_invariant_crop_mask_index=1,
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis"
            ),
        )
    )

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        GaussianBlurTransform(
            (0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5,
        )
    )
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.75, 1.25), p_per_sample=0.15
        )
    )

    if params.get("do_additive_brightness"):
        tr_transforms.append(
            BrightnessTransform(
                params.get("additive_brightness_mu"),
                params.get("additive_brightness_sigma"),
                True,
                p_per_sample=params.get("additive_brightness_p_per_sample"),
                p_per_channel=params.get("additive_brightness_p_per_channel"),
            )
        )

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=ignore_axes,
        )
    )
    tr_transforms.append(
        GammaTransform(
            params.get("gamma_range"),
            True,
            True,
            retain_stats=params.get("gamma_retain_stats"),
            p_per_sample=0.1,
        )
    )  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"],
            )
        )

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    tr_transforms.append(
        Make25dInput(
            slice_per_25d_group=params.get("slice_per_25d_group"),
            num_25d_group=params.get("num_25d_group"),
            group_25d_overlap=params.get("group_25d_overlap"),
        )
    )

    tr_transforms.append(NumpyToTensor(["data", "mask"], "float"))

    return tr_transforms


def get_rtnet_25d_dual_transforms(
    params,
    final_patch_size,
    patch_center_dist_from_border,
    order_data,
    border_val_mask,
    order_mask,
):
    tr_transforms = []

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = final_patch_size[1:]
    else:
        patch_size_spatial = final_patch_size
        ignore_axes = None

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=patch_center_dist_from_border,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=order_data,
            border_mode_mask="constant",
            border_cval_mask=border_val_mask,
            order_mask=order_mask,
            random_crop=params.get("random_crop"),
            size_invariant_crop=True,
            size_invariant_crop_mask_index=1,
            copy_and_preserve_original=True,
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis"
            ),
        )
    )

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        GaussianBlurTransform(
            (0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5,
        )
    )
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.75, 1.25), p_per_sample=0.15
        )
    )

    if params.get("do_additive_brightness"):
        tr_transforms.append(
            BrightnessTransform(
                params.get("additive_brightness_mu"),
                params.get("additive_brightness_sigma"),
                True,
                p_per_sample=params.get("additive_brightness_p_per_sample"),
                p_per_channel=params.get("additive_brightness_p_per_channel"),
            )
        )

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=ignore_axes,
        )
    )
    # tr_transforms.append(
    #     GammaTransform(params.get("gamma_range"),
    #                    True,
    #                    True,
    #                    retain_stats=params.get("gamma_retain_stats"),
    #                    p_per_sample=0.1))  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"],
            )
        )

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    tr_transforms.append(
        Make25dInput(
            slice_per_25d_group=params.get("slice_per_25d_group"),
            num_25d_group=params.get("num_25d_group"),
            group_25d_overlap=params.get("group_25d_overlap"),
        )
    )

    tr_transforms.append(NumpyToTensor(["data", "mask"], "float"))

    return tr_transforms


def get_rtnet_size_invariant_transforms(
    params,
    final_patch_size,
    patch_center_dist_from_border,
    order_data,
    border_val_mask,
    order_mask,
):
    tr_transforms = []

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = final_patch_size[1:]
    else:
        patch_size_spatial = final_patch_size
        ignore_axes = None

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=patch_center_dist_from_border,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=order_data,
            border_mode_mask="constant",
            border_cval_mask=border_val_mask,
            order_mask=order_mask,
            random_crop=params.get("random_crop"),
            size_invariant_crop=True,
            size_invariant_crop_mask_index=1,
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis"
            ),
        )
    )

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        GaussianBlurTransform(
            (0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5,
        )
    )
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.75, 1.25), p_per_sample=0.15
        )
    )

    if params.get("do_additive_brightness"):
        tr_transforms.append(
            BrightnessTransform(
                params.get("additive_brightness_mu"),
                params.get("additive_brightness_sigma"),
                True,
                p_per_sample=params.get("additive_brightness_p_per_sample"),
                p_per_channel=params.get("additive_brightness_p_per_channel"),
            )
        )

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=ignore_axes,
        )
    )
    tr_transforms.append(
        GammaTransform(
            params.get("gamma_range"),
            True,
            True,
            retain_stats=params.get("gamma_retain_stats"),
            p_per_sample=0.1,
        )
    )  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"],
            )
        )

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if len(patch_size_spatial) == 2:
        tr_transforms.append(MakePseudoRGBInput())

    tr_transforms.append(NumpyToTensor(["data", "mask"], "float"))

    return tr_transforms
