import random

import math
import numpy as np

from .abstract_transforms import AbstractTransform
from .utils import (
    create_zero_centered_coordinate_mesh,
    get_mask_crop_center,
    interpolate_img,
    scale_coords,
)


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict["data"].shape
    data_dict["data"] = data_dict["data"].reshape(
        (shp[0], shp[1] * shp[2], shp[3], shp[4])
    )
    data_dict["orig_shape_data"] = shp
    shp = data_dict["seg"].shape
    data_dict["seg"] = data_dict["seg"].reshape(
        (shp[0], shp[1] * shp[2], shp[3], shp[4])
    )
    data_dict["orig_shape_seg"] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict["orig_shape_data"]
    current_shape = data_dict["data"].shape
    data_dict["data"] = data_dict["data"].reshape(
        (shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1])
    )
    shp = data_dict["orig_shape_seg"]
    current_shape_seg = data_dict["seg"].shape
    data_dict["seg"] = data_dict["seg"].reshape(
        (shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1])
    )
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class Make25dInput(AbstractTransform):
    def __init__(
        self, slice_per_25d_group=3, num_25d_group=1, group_25d_overlap=0, is_test=False
    ):
        self.slice_per_group = slice_per_25d_group
        self.num_group = num_25d_group
        self.group_slice_overlap = group_25d_overlap
        self.is_test = is_test

    def __call__(self, **data_dict):
        image_data = data_dict["data"]
        mask_data = data_dict["mask"]

        nonzero_slice_idx = np.unique(np.nonzero(mask_data[0, 0])[0]).tolist()
        nonzero_slice_idx.sort()
        total_nonzero_slice = len(nonzero_slice_idx)
        total_target_slice = (
            self.slice_per_group * self.num_group
            - self.group_slice_overlap * (self.num_group - 1)
        )

        # if there are sufficient fgr slices, then select slices from an eligible
        # random start position, else copy the first and last slices to fill the gap
        if total_nonzero_slice >= total_target_slice:
            if not self.is_test:
                # random sample the start position during training
                src_start = random.randint(0, total_nonzero_slice - total_target_slice)
                src_end = src_start + total_target_slice
                src_slices = nonzero_slice_idx[src_start:src_end]
            else:
                shift = (total_nonzero_slice - total_target_slice) // 2
                src_slices = nonzero_slice_idx[shift : shift + total_target_slice]
        else:
            if len(nonzero_slice_idx) == 0:
                src_slices = [i for i in range(total_target_slice)]
            else:
                diff = total_target_slice - total_nonzero_slice
                pad_up = [nonzero_slice_idx[0]] * (diff // 2)
                pad_down = [nonzero_slice_idx[-1]] * (diff - diff // 2)
                src_slices = pad_up + nonzero_slice_idx + pad_down

        image_out, mask_out = [], []
        for g_idx in range(self.num_group):
            start_idx = g_idx * (self.slice_per_group - self.group_slice_overlap)
            end_idx = start_idx + self.slice_per_group
            image_out.append(image_data[:, :, src_slices[start_idx:end_idx]])
            mask_out.append(mask_data[:, :, src_slices[start_idx:end_idx]])
        data_dict["data"] = np.concatenate(image_out, axis=2).transpose(
            1, 0, 2, 3, 4
        )  # (mod, c, d, h, w)
        data_dict["mask"] = np.concatenate(mask_out, axis=2).transpose(
            1, 0, 2, 3, 4
        )  # (mod, c, d, h, w)

        return data_dict


class MakePseudoRGBInput(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        single_channel_image = data_dict["data"]
        single_channel_mask = data_dict["mask"]
        # shp = single_channel_image.shape
        # single_channel_image = single_channel_image.reshape(1, *shp[-2:])
        # single_channel_mask = single_channel_mask.reshape(1, *shp[-2:])
        pseudo_rgb_image = np.vstack([single_channel_image] * 3).transpose(1, 0, 2, 3)
        pseudo_rgb_mask = np.vstack([single_channel_mask] * 3).transpose(1, 0, 2, 3)
        data_dict["data"] = pseudo_rgb_image
        # data_dict['data'] = pseudo_rgb_image[None, ...]
        data_dict["mask"] = pseudo_rgb_mask
        # data_dict['mask'] = pseudo_rgb_mask[None, ...]
        return data_dict


class MakePseudoRGBInputFrom3D(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        single_channel_image = data_dict["data"]
        single_channel_mask = data_dict["mask"]
        # shp = single_channel_image.shape
        # single_channel_image = single_channel_image.reshape(1, *shp[-3:])
        # single_channel_mask = single_channel_mask.reshape(1, *shp[-3:])
        pseudo_rgb_image = np.concatenate([single_channel_image] * 3, axis=0).transpose(
            1, 0, 2, 3, 4
        )
        pseudo_rgb_mask = np.concatenate([single_channel_mask] * 3, axis=0).transpose(
            1, 0, 2, 3, 4
        )
        data_dict["data"] = pseudo_rgb_image
        data_dict["mask"] = pseudo_rgb_mask
        return data_dict


class PersliceInterpolateAndCrop(AbstractTransform):
    def __init__(
        self,
        final_patch_size,
        data_key="data",
        label_key="mask",
        border_mode_data="constant",
        border_cval_data=0,
        order_data=3,
        border_mode_mask="constant",
        border_cval_mask=-1,
        order_mask=1,
        drop_top_and_bottom=False,
    ):
        self.final_patch_size = final_patch_size
        self.data_key = data_key
        self.label_key = label_key
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_mask = border_mode_mask
        self.border_cval_mask = border_cval_mask
        self.order_mask = order_mask
        self.drop_top_and_bottom = drop_top_and_bottom

    def __call__(self, **data_dict):
        raw_data = data_dict.get(self.data_key)  # (mod, chn, z, x, y)
        raw_mask = data_dict.get(self.label_key)
        dim = len(raw_data.shape)
        assert dim == 5, "The dimension must be 5, but {} got".format(dim)

        # find nonzero mask layers
        nonzero_layers = np.unique(np.where(raw_mask[0][0] > 0)[0]).tolist()
        if self.drop_top_and_bottom and len(nonzero_layers) > 3:
            nonzero_layers = nonzero_layers[1:-1]

        mask_result = np.zeros(
            (
                raw_mask.shape[0],
                raw_mask.shape[1],
                self.final_patch_size[0],
                self.final_patch_size[1],
                self.final_patch_size[2],
            ),
            dtype=np.float32,
        )

        data_result = np.zeros(
            (
                raw_data.shape[0],
                raw_data.shape[1],
                self.final_patch_size[0],
                self.final_patch_size[1],
                self.final_patch_size[2],
            ),
            dtype=np.float32,
        )

        layer_offset = (self.final_patch_size[0] - len(nonzero_layers)) // 2
        assert layer_offset >= 0, "Layer offset must be a non-negative number"

        for layer_idx, layer_id in enumerate(nonzero_layers):
            # get mask crop center and override the original fixed patch size
            mask_ctr, box_size = get_mask_crop_center(raw_mask[0][1][layer_id])
            # A hack way to deal with the 2D/3D situation, need modify in the future
            mask_ctr = mask_ctr[-2:]
            box_size = box_size[-2:]
            scale_ratio = (
                max(box_size) * 1.0 / max(self.final_patch_size[-2:])
            )  # spatial ratio

            for sample_id in range(raw_data.shape[0]):
                coords = create_zero_centered_coordinate_mesh(
                    self.final_patch_size[-2:]
                )

                sc = [scale_ratio, scale_ratio]
                coords = scale_coords(coords, sc)

                for d in range(2):
                    ctr = mask_ctr[d]
                    coords[d] += ctr

                for channel_id in range(raw_data.shape[1]):
                    data_result[
                        sample_id, channel_id, layer_offset + layer_idx
                    ] = interpolate_img(
                        raw_data[sample_id, channel_id, layer_id],
                        coords,
                        self.order_data,
                        self.border_mode_data,
                        cval=self.border_cval_data,
                    )

                for channel_id in range(raw_mask.shape[1]):
                    mask_result[
                        sample_id, channel_id, layer_offset + layer_idx
                    ] = interpolate_img(
                        raw_mask[sample_id, channel_id, layer_id],
                        coords,
                        self.order_mask,
                        self.border_mode_mask,
                        cval=self.border_cval_mask,
                        is_mask=True,
                    )

        return {self.data_key: data_result, self.label_key: mask_result}


class SizeInvariantInterpolateAndCrop(AbstractTransform):
    def __init__(
        self,
        final_patch_size,
        data_key="data",
        label_key="mask",
        border_mode_data="constant",
        border_cval_data=0,
        order_data=3,
        border_mode_mask="constant",
        border_cval_mask=-1,
        order_mask=1,
        mask_index=0,
    ):
        self.final_patch_size = final_patch_size
        self.data_key = data_key
        self.label_key = label_key
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_mask = border_mode_mask
        self.border_cval_mask = border_cval_mask
        self.order_mask = order_mask
        self.mask_index = mask_index

    def __call__(self, **data_dict):
        raw_data = data_dict.get(self.data_key)  # (mod, chn, z, x, y)
        raw_mask = data_dict.get(self.label_key)

        dim = len(self.final_patch_size)
        assert dim == 3, "The volumetric dimension must be 3, but {} got".format(dim)

        mask_result = np.zeros(
            (
                raw_mask.shape[0],
                raw_mask.shape[1],
                self.final_patch_size[0],
                self.final_patch_size[1],
                self.final_patch_size[2],
            ),
            dtype=np.float32,
        )
        data_result = np.zeros(
            (
                raw_data.shape[0],
                raw_data.shape[1],
                self.final_patch_size[0],
                self.final_patch_size[1],
                self.final_patch_size[2],
            ),
            dtype=np.float32,
        )

        mask_ctr, box_size = get_mask_crop_center(raw_mask[0][self.mask_index])
        # A hack way to deal with the 2D/3D situation, need modify in the future
        mask_ctr = mask_ctr[-len(self.final_patch_size) :]
        box_size = box_size[-len(self.final_patch_size) :]
        scale_ratio = (
            max(box_size[-2:]) * 1.0 / max(self.final_patch_size[-2:])
        )  # spatial ratio
        sc = [1, scale_ratio, scale_ratio]

        for sample_id in range(raw_data.shape[0]):
            coords = create_zero_centered_coordinate_mesh(self.final_patch_size)
            coords = scale_coords(coords, sc)
            for d in range(dim):
                # ctr = mask_ctr[d]
                ctr = raw_data.shape[d + 2] / 2.0 - 0.5
                coords[d] += ctr

            for channel_id in range(raw_data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(
                    raw_data[sample_id, channel_id],
                    coords,
                    self.order_data,
                    self.border_mode_data,
                    cval=self.border_cval_data,
                )
            if raw_mask is not None:
                for channel_id in range(raw_mask.shape[1]):
                    mask_result[sample_id, channel_id] = interpolate_img(
                        raw_mask[sample_id, channel_id],
                        coords,
                        self.order_mask,
                        self.border_mode_mask,
                        cval=self.border_cval_mask,
                        is_mask=True,
                    )

        return {self.data_key: data_result, self.label_key: mask_result}


class DualInputInterpolateAndCrop(AbstractTransform):
    def __init__(
        self,
        final_patch_size,
        data_key="data",
        label_key="mask",
        border_mode_data="constant",
        border_cval_data=0,
        order_data=3,
        border_mode_mask="constant",
        border_cval_mask=-1,
        order_mask=1,
        mask_index=0,
    ):
        self.final_patch_size = final_patch_size
        self.data_key = data_key
        self.label_key = label_key
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_mask = border_mode_mask
        self.border_cval_mask = border_cval_mask
        self.order_mask = order_mask
        self.mask_index = mask_index

    def __call__(self, **data_dict):
        raw_data = data_dict.get(self.data_key)  # (mod, chn, z, x, y)
        raw_data = np.concatenate((raw_data, np.copy(raw_data)), axis=1)
        raw_mask = data_dict.get(self.label_key)
        raw_mask = np.concatenate((raw_mask, np.copy(raw_mask)), axis=1)

        dim = len(self.final_patch_size)
        assert dim == 3, "The volumetric dimension must be 3, but {} got".format(dim)

        mask_result = np.zeros(
            (
                raw_mask.shape[0],
                raw_mask.shape[1],
                self.final_patch_size[0],
                self.final_patch_size[1],
                self.final_patch_size[2],
            ),
            dtype=np.float32,
        )
        data_result = np.zeros(
            (
                raw_data.shape[0],
                raw_data.shape[1],
                self.final_patch_size[0],
                self.final_patch_size[1],
                self.final_patch_size[2],
            ),
            dtype=np.float32,
        )

        if np.max(raw_mask) == 0:
            # print("Warning: empty mask, return all zero array")
            return {self.data_key: data_result, self.label_key: mask_result}

        mask_ctr, box_size = get_mask_crop_center(raw_mask[0][self.mask_index])
        # A hack way to deal with the 2D/3D situation, need modify in the future
        mask_ctr = mask_ctr[-len(self.final_patch_size) :]
        box_size = box_size[-len(self.final_patch_size) :]
        scale_ratio = (
            max(box_size[-2:]) * 1.0 / max(self.final_patch_size[-2:])
        )  # spatial ratio
        sc = [1, scale_ratio, scale_ratio]

        for sample_id in range(raw_data.shape[0]):
            coords = create_zero_centered_coordinate_mesh(self.final_patch_size)
            coords = np.stack(
                [coords] * raw_data.shape[1], axis=0
            )  # duplicate for preserving/invariant

            coords[-1] = scale_coords(coords[-1], sc)
            for d in range(dim):
                # ctr = mask_ctr[d]
                ctr = raw_data.shape[d + 2] / 2.0 - 0.5
                for c in range(coords.shape[0]):
                    coords[c][d] += ctr

            for channel_id in range(raw_data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(
                    raw_data[sample_id, channel_id],
                    coords[channel_id],
                    self.order_data,
                    self.border_mode_data,
                    cval=self.border_cval_data,
                )
            if raw_mask is not None:
                for channel_id in range(raw_mask.shape[1]):
                    mask_result[sample_id, channel_id] = interpolate_img(
                        raw_mask[sample_id, channel_id],
                        coords[math.trunc(channel_id / 2)],
                        self.order_mask,
                        self.border_mode_mask,
                        cval=self.border_cval_mask,
                        is_mask=True,
                    )

        return {self.data_key: data_result, self.label_key: mask_result}
