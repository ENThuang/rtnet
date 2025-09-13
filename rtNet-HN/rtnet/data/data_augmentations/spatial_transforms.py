import numpy as np

from .abstract_transforms import AbstractTransform
from .spatial_augmentations import augment_mirroring, augment_spatial


class MirrorTransform(AbstractTransform):
    """Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(
        self, axes=(0, 1, 2), data_key="data", label_key="mask", p_per_sample=1
    ):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError(
                "MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                "is now axes=(0, 1, 2). Please adapt your scripts accordingly."
            )

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        mask = data_dict.get(self.label_key)

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                sample_mask = None
                if mask is not None:
                    sample_mask = mask[b]
                ret_val = augment_mirroring(data[b], sample_mask, axes=self.axes)
                data[b] = ret_val[0]
                if mask is not None:
                    mask[b] = ret_val[1]

        data_dict[self.data_key] = data
        if mask is not None:
            data_dict[self.label_key] = mask

        return data_dict


class SpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_mask: How to treat border pixels in mask? see scipy.ndimage.map_coordinates

        border_cval_mask: If border_mode_mask=constant, what value to use?

        order_mask: Order of interpolation for mask. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (e.g.,  if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(
        self,
        patch_size,
        patch_center_dist_from_border=30,
        do_elastic_deform=True,
        alpha=(0.0, 1000.0),
        sigma=(10.0, 13.0),
        do_rotation=True,
        angle_x=(0, 2 * np.pi),
        angle_y=(0, 2 * np.pi),
        angle_z=(0, 2 * np.pi),
        do_scale=True,
        scale=(0.75, 1.25),
        border_mode_data="nearest",
        border_cval_data=0,
        order_data=3,
        border_mode_mask="constant",
        border_cval_mask=0,
        order_mask=0,
        random_crop=True,
        size_invariant_crop=False,
        size_invariant_crop_mask_index=0,
        copy_and_preserve_original=False,
        data_key="data",
        label_key="mask",
        p_el_per_sample=1,
        p_scale_per_sample=1,
        p_rot_per_sample=1,
        independent_scale_for_each_axis=False,
        p_rot_per_axis: float = 1,
        p_independent_scale_per_axis: int = 1,
    ):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_mask = border_mode_mask
        self.border_cval_mask = border_cval_mask
        self.order_mask = order_mask
        self.random_crop = random_crop
        self.size_invariant_crop = size_invariant_crop
        self.size_invariant_crop_mask_index = size_invariant_crop_mask_index
        self.copy_and_preserve_original = copy_and_preserve_original
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        mask = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(
            data,
            mask,
            patch_size=patch_size,
            patch_center_dist_from_border=self.patch_center_dist_from_border,
            do_elastic_deform=self.do_elastic_deform,
            alpha=self.alpha,
            sigma=self.sigma,
            do_rotation=self.do_rotation,
            angle_x=self.angle_x,
            angle_y=self.angle_y,
            angle_z=self.angle_z,
            do_scale=self.do_scale,
            scale=self.scale,
            border_mode_data=self.border_mode_data,
            border_cval_data=self.border_cval_data,
            order_data=self.order_data,
            border_mode_mask=self.border_mode_mask,
            border_cval_mask=self.border_cval_mask,
            order_mask=self.order_mask,
            random_crop=self.random_crop,
            size_invariant_crop=self.size_invariant_crop,
            size_invariant_crop_mask_index=self.size_invariant_crop_mask_index,
            copy_and_preserve_original=self.copy_and_preserve_original,
            p_el_per_sample=self.p_el_per_sample,
            p_scale_per_sample=self.p_scale_per_sample,
            p_rot_per_sample=self.p_rot_per_sample,
            independent_scale_for_each_axis=self.independent_scale_for_each_axis,
            p_rot_per_axis=self.p_rot_per_axis,
            p_independent_scale_per_axis=self.p_independent_scale_per_axis,
        )
        data_dict[self.data_key] = ret_val[0]
        if mask is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict
