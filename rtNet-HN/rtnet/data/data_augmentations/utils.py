import random
import numpy as np
from copy import deepcopy
from scipy.ndimage import map_coordinates, fourier_gaussian
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.morphology import grey_dilation
from skimage.transform import resize
from scipy.ndimage.measurements import label as lb

from rtnet.utils.utils import find_bbox_from_mask


def uniform(low, high, size=None):
    """
    wrapper for np.random.uniform to allow it to handle low=high
    :param low:
    :param high:
    :return:
    """
    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return np.random.uniform(low, high, size)


def mask_random_square(
    img, square_size, n_val, channel_wise_n_val=False, square_pos=None
):
    """Masks (sets = 0) a random square in an image"""

    img_h = img.shape[-2]
    img_w = img.shape[-1]

    img = img.copy()

    if square_pos is None:
        w_start = np.random.randint(0, img_w - square_size)
        h_start = np.random.randint(0, img_h - square_size)
    else:
        pos_wh = square_pos[np.random.randint(0, len(square_pos))]
        w_start = pos_wh[0]
        h_start = pos_wh[1]

    if img.ndim == 2:
        rnd_n_val = get_range_val(n_val)
        img[
            h_start : (h_start + square_size), w_start : (w_start + square_size)
        ] = rnd_n_val
    elif img.ndim == 3:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = get_range_val(n_val)
                img[
                    i,
                    h_start : (h_start + square_size),
                    w_start : (w_start + square_size),
                ] = rnd_n_val
        else:
            rnd_n_val = get_range_val(n_val)
            img[
                :, h_start : (h_start + square_size), w_start : (w_start + square_size)
            ] = rnd_n_val
    elif img.ndim == 4:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = get_range_val(n_val)
                img[
                    :,
                    i,
                    h_start : (h_start + square_size),
                    w_start : (w_start + square_size),
                ] = rnd_n_val
        else:
            rnd_n_val = get_range_val(n_val)
            img[
                :,
                :,
                h_start : (h_start + square_size),
                w_start : (w_start + square_size),
            ] = rnd_n_val

    return img


def mask_random_squares(
    img, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None
):
    """Masks a given number of squares in an image"""
    for i in range(n_squares):
        img = mask_random_square(
            img,
            square_size,
            n_val,
            channel_wise_n_val=channel_wise_n_val,
            square_pos=square_pos,
        )
    return img


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError(
                "value must be either a single value or a list/tuple of len 2"
            )
        return n_val
    else:
        return value


def illumination_jitter(img, u, s, sigma):
    # img must have shape [....., c] where c is the color channel
    alpha = np.random.normal(0, sigma, s.shape)
    jitter = np.dot(u, alpha * s)
    img2 = np.array(img)
    for c in range(img.shape[0]):
        img2[c] = img[c] + jitter[c]
    return img2


def general_cc_var_num_channels(
    img,
    diff_order=0,
    mink_norm=1,
    sigma=1,
    mask_im=None,
    saturation_threshold=255,
    dilation_size=3,
    clip_range=True,
):
    # img must have first dim color channel! img[c, x, y(, z, ...)]
    dim_img = len(img.shape[1:])
    if clip_range:
        minm = img.min()
        maxm = img.max()
    img_internal = np.array(img)
    if mask_im is None:
        mask_im = np.zeros(img_internal.shape[1:], dtype=bool)
    img_dil = deepcopy(img_internal)
    for c in range(img.shape[0]):
        img_dil[c] = grey_dilation(img_internal[c], tuple([dilation_size] * dim_img))
    mask_im = mask_im | np.any(img_dil >= saturation_threshold, axis=0)
    if sigma != 0:
        mask_im[:sigma, :] = 1
        mask_im[mask_im.shape[0] - sigma :, :] = 1
        mask_im[:, mask_im.shape[1] - sigma :] = 1
        mask_im[:, :sigma] = 1
        if dim_img == 3:
            mask_im[:, :, mask_im.shape[2] - sigma :] = 1
            mask_im[:, :, :sigma] = 1

    output_img = deepcopy(img_internal)

    if diff_order == 0 and sigma != 0:
        for c in range(img_internal.shape[0]):
            img_internal[c] = gaussian_filter(img_internal[c], sigma, diff_order)
    elif diff_order == 1:
        for c in range(img_internal.shape[0]):
            img_internal[c] = gaussian_gradient_magnitude(img_internal[c], sigma)
    elif diff_order > 1:
        raise ValueError(
            "diff_order can only be 0 or 1. 2 is not supported (ToDo, maybe)"
        )

    img_internal = np.abs(img_internal)

    white_colors = []

    if mink_norm != -1:
        kleur = np.power(img_internal, mink_norm)
        for c in range(kleur.shape[0]):
            white_colors.append(
                np.power((kleur[c][mask_im != 1]).sum(), 1.0 / mink_norm)
            )
    else:
        for c in range(img_internal.shape[0]):
            white_colors.append(np.max(img_internal[c][mask_im != 1]))

    som = np.sqrt(np.sum([i**2 for i in white_colors]))

    white_colors = [i / som for i in white_colors]

    for c in range(output_img.shape[0]):
        output_img[c] /= white_colors[c] * np.sqrt(3.0)

    if clip_range:
        output_img[output_img < minm] = minm
        output_img[output_img > maxm] = maxm
    return white_colors, output_img


def generate_elastic_transform_coordinates(shape, alpha, sigma):
    n_dim = len(shape)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter(
                (np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.meshgrid(*tmp, indexing="ij")
    indices = [np.reshape(i + j, (-1, 1)) for i, j in zip(offsets, coords)]
    return indices


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing="ij")).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.0)[d]
    return coords


def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter(
                (np.random.random(coordinates.shape[1:]) * 2 - 1),
                sigma,
                mode="constant",
                cval=0,
            )
            * alpha
        )
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = (
        np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix)
        .transpose()
        .reshape(coords.shape)
    )
    return coords


def create_matrix_rotation_2d(angle, matrix=None):
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords = (
        np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix)
        .transpose()
        .reshape(coords.shape)
    )
    return coords


def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords


def interpolate_img(img, coords, order=3, mode="nearest", cval=0.0, is_mask=False):
    if is_mask and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates(
                (img == c).astype(float), coords, order=order, mode=mode, cval=cval
            )
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(
            img.astype(float), coords, order=order, mode=mode, cval=cval
        ).astype(img.dtype)


def get_lbs_for_random_crop(crop_size, data_shape, margins):
    """

    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :param margins:
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        if data_shape[i + 2] - crop_size[i] - margins[i] > margins[i]:
            lbs.append(
                np.random.randint(
                    margins[i], data_shape[i + 2] - crop_size[i] - margins[i]
                )
            )
        else:
            lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def get_lbs_for_center_crop(crop_size, data_shape):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def crop(
    data,
    seg=None,
    crop_size=128,
    margins=(0, 0, 0),
    crop_type="center",
    pad_mode="constant",
    pad_kwargs={"constant_values": 0},
    pad_mode_seg="constant",
    pad_kwargs_seg={"constant_values": 0},
):
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes

    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), (
            "data and seg must have the same spatial "
            "dimensions. Data: %s, seg: %s" % (str(data_shape), str(seg_shape))
        )

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(data_shape) - 2, (
            "If you provide a list/tuple as center crop make sure it has the same dimension as your "
            "data (2d/3d)"
        )

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros(
        [data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype
    )
    if seg is not None:
        seg_return = np.zeros(
            [seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype
        )
    else:
        seg_return = None

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        if seg is not None:
            seg_shape_here = [seg_shape[0]] + list(seg[b].shape)

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError("crop_type must be either center or random")

        need_to_pad = [[0, 0]] + [
            [
                abs(min(0, lbs[d])),
                abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d]))),
            ]
            for d in range(dim)
        ]

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d + 2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [
            slice(lbs[d], ubs[d]) for d in range(dim)
        ]
        data_cropped = data[b][tuple(slicer_data)]

        if seg_return is not None:
            slicer_seg = [slice(0, seg_shape_here[1])] + [
                slice(lbs[d], ubs[d]) for d in range(dim)
            ]
            seg_cropped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(
                    seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg
                )
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

    return data_return, seg_return


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    return crop(data, seg, crop_size, margins, "random")


def center_crop(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 0, "center")


def get_mask_crop_center(mask):
    dim = len(mask.shape)
    if dim == 3:
        x_min, y_min, z_min, h, w, d = find_bbox_from_mask(mask)
        ctr_x = int(x_min + h / 2.0 - 0.5)
        ctr_y = int(y_min + w / 2.0 - 0.5)
        ctr_z = int(z_min + d / 2.0 - 0.5)
        return (ctr_x, ctr_y, ctr_z), (h, w, d)
    elif dim == 2:
        x_min, y_min, h, w = find_bbox_from_mask(mask)
        ctr_x = int(x_min + h / 2.0 - 0.5)
        ctr_y = int(y_min + w / 2.0 - 0.5)
        return (ctr_x, ctr_y), (h, w)
    else:
        raise ValueError
