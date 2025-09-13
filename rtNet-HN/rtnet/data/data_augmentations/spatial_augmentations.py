from builtins import range

import numpy as np
import math
from .utils import (
    create_zero_centered_coordinate_mesh,
    elastic_deform_coordinates,
    rotate_coords_3d,
    rotate_coords_2d,
    scale_coords,
    interpolate_img,
    get_mask_crop_center,
)
from .utils import random_crop as random_crop_aug
from .utils import center_crop as center_crop_aug


def augment_spatial(
    data,
    mask,
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
    p_el_per_sample=1,
    p_scale_per_sample=1,
    p_rot_per_sample=1,
    independent_scale_for_each_axis=False,
    p_rot_per_axis: float = 1,
    p_independent_scale_per_axis: int = 1,
):
    # first check if the mask is empty (caused by z-axis interpolate)
    if np.max(mask) == 0:
        # just return an all zero array with the target shape
        data = np.concatenate((data, np.copy(data)), axis=1)
        mask = np.concatenate((mask, np.copy(mask)), axis=1)
        data_result = np.ones((*data.shape[: -len(patch_size)], *patch_size)).astype(
            np.float32
        )
        mask_result = np.zeros((*mask.shape[: -len(patch_size)], *patch_size)).astype(
            np.float32
        )
        # print("Warning: empty mask, return all zero array")
        return data_result, mask_result

    # duplicate data and mask for size preserving and invariant DualNet
    # assume channel 0 is the original, channel 1 is the size-invariant data
    if copy_and_preserve_original:
        data = np.concatenate((data, np.copy(data)), axis=1)
        mask = np.concatenate((mask, np.copy(mask)), axis=1)

    # get mask crop center and override the original fixed patch size
    if size_invariant_crop:
        mask_ctr, box_size = get_mask_crop_center(
            mask[0][size_invariant_crop_mask_index]
        )
        # A hack way to deal with the 2D/3D situation, need modify in the future
        mask_ctr = mask_ctr[-len(patch_size) :]
        box_size = box_size[-len(patch_size) :]
        scale_ratio = max(box_size[-2:]) * 1.0 / max(patch_size[-2:])  # spatial ratio

    dim = len(patch_size)
    mask_result = None
    if mask is not None:
        if dim == 2:
            mask_result = np.zeros(
                (mask.shape[0], mask.shape[1], patch_size[0], patch_size[1]),
                dtype=np.float32,
            )
        else:
            mask_result = np.zeros(
                (
                    mask.shape[0],
                    mask.shape[1],
                    patch_size[0],
                    patch_size[1],
                    patch_size[2],
                ),
                dtype=np.float32,
            )

    if dim == 2:
        data_result = np.zeros(
            (data.shape[0], data.shape[1], patch_size[0], patch_size[1]),
            dtype=np.float32,
        )
    else:
        data_result = np.zeros(
            (data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
            dtype=np.float32,
        )

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        if copy_and_preserve_original:
            coords = np.stack(
                [coords] * data.shape[1], axis=0
            )  # duplicate for preserving/invariant
        else:
            coords = coords[None, ...]
        modified_coords = False

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            for c in range(coords.shape[0]):
                coords[c] = elastic_deform_coordinates(coords[c], a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:
            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0
                for c in range(coords.shape[0]):
                    coords[c] = rotate_coords_3d(coords[c], a_x, a_y, a_z)
            else:
                for c in range(coords.shape[0]):
                    coords[c] = rotate_coords_2d(coords[c], a_x)
            modified_coords = True

        if size_invariant_crop:  # scale the ROI spatial size to match the target size
            if dim == 3:
                sc = [1, scale_ratio, scale_ratio]
            elif dim == 2:
                sc = [scale_ratio, scale_ratio]
            else:
                raise ValueError
            coords[-1] = scale_coords(coords[-1], sc)  # only do it for the last channel
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if (
                independent_scale_for_each_axis
                and np.random.uniform() < p_independent_scale_per_axis
            ):
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])
            # for c in range(coords.shape[0]):
            #     coords[c] = scale_coords(coords[c], sc)
            if not size_invariant_crop or copy_and_preserve_original:
                coords[0] = scale_coords(
                    coords[0], sc
                )  # only do scale for presv channel
                modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(
                        patch_center_dist_from_border[d],
                        data.shape[d + 2] - patch_center_dist_from_border[d],
                    )
                # elif size_invariant_crop:
                #     ctr = mask_ctr[d]
                else:
                    ctr = data.shape[d + 2] / 2.0 - 0.5
                for c in range(coords.shape[0]):
                    coords[c][d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(
                    data[sample_id, channel_id],
                    coords[channel_id],
                    order_data,
                    border_mode_data,
                    cval=border_cval_data,
                )
            if mask is not None:
                for channel_id in range(mask.shape[1]):
                    mask_result[sample_id, channel_id] = interpolate_img(
                        mask[sample_id, channel_id],
                        coords[math.trunc(channel_id / 2)],
                        order_mask,
                        border_mode_mask,
                        cval=border_cval_mask,
                        is_mask=True,
                    )
        else:
            if mask is None:
                s = None
            else:
                s = mask[sample_id : sample_id + 1]
            if random_crop:
                margin = [
                    patch_center_dist_from_border[d] - patch_size[d] // 2
                    for d in range(dim)
                ]
                d, s = random_crop_aug(
                    data[sample_id : sample_id + 1], s, patch_size, margin
                )
            else:
                d, s = center_crop_aug(data[sample_id : sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if mask is not None:
                mask_result[sample_id] = s[0]
    return data_result, mask_result


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]"
        )
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg
