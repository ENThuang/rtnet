import os
import collections
from typing import Any, Tuple, List, Sequence
from itertools import repeat
import importlib
import pkgutil
import numpy as np


def get_center_of_mass(data):
    assert len(data.shape) == 3, "Only support 3D data."
    nonzero_coords = np.argwhere(data != 0)
    ctr_x = np.mean(nonzero_coords[..., 0])
    ctr_y = np.mean(nonzero_coords[..., 1])
    ctr_z = np.mean(nonzero_coords[..., 2])
    return (ctr_x, ctr_y, ctr_z)


def crop_and_pad(
    input,
    full_size,
    box_center,
    crop_size,
    ccl=None,
    ins_label=None,
    remove_adjacent_lymph_nodes=False,
    pad_mode="constant",
    pad_kwargs_data={},
):
    """_summary_

    _extended_summary_

    Args:
        input (_type_): _description_
        full_size (_type_): _description_
        box_center (_type_): _description_
        crop_size (_type_): _description_
        ccl (_type_, optional): _description_. Defaults to None.
        ins_label (_type_, optional): _description_. Defaults to None.
        remove_adjacent_lymph_nodes (bool, optional): _description_. Defaults to False.

    Returns:
        Describe the type and semantics of the return value. If the function only returns None, this section is not required.
        If exists multiple return values with individual names, describe such a return value as: “Returns: A tuple (mat_a, mat_b), where mat_a is …, and …”.
    """
    dim = len(input.shape)
    crop_src = np.copy(input)

    if dim == 3:
        crop_x_lb, crop_y_lb, crop_z_lb = box_center - np.array(crop_size) // 2
        crop_x_ub, crop_y_ub, crop_z_ub = box_center + np.array(crop_size) // 2
    else:
        crop_x_lb, crop_y_lb = box_center - np.array(crop_size) // 2
        crop_x_ub, crop_y_ub = box_center + np.array(crop_size) // 2

    # crop within the boundary
    valid_bbox_x_lb = max(0, crop_x_lb)
    valid_bbox_x_ub = min(full_size[0], crop_x_ub)
    valid_bbox_y_lb = max(0, crop_y_lb)
    valid_bbox_y_ub = min(full_size[1], crop_y_ub)

    if dim == 3:
        valid_bbox_z_lb = max(0, crop_z_lb)
        valid_bbox_z_ub = min(full_size[2], crop_z_ub)
        cropped = crop_src[
            valid_bbox_x_lb:valid_bbox_x_ub,
            valid_bbox_y_lb:valid_bbox_y_ub,
            valid_bbox_z_lb:valid_bbox_z_ub,
        ]
    else:
        cropped = crop_src[
            valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub
        ]

    # check if the cropped region includes other instance
    # if remove_adjacent_lymph_nodes:
    #     assert ccl is not None and ins_label is not None, (
    #         "Connect components mask 'ccl' and the current instance label 'ins_label' is required")
    #     if len(ccl.shape) == 3:
    #         ccl = np.expand_dims(ccl, axis=0)
    #     instance_map = np.copy(ccl[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub,
    #                                valid_bbox_z_lb:valid_bbox_z_ub])
    #     num_instance = len(np.unique(instance_map[instance_map != 0]))
    #     if num_instance > 1:
    #         cropped[np.logical_and(instance_map != ins_label, instance_map != 0)] = -1024

    # pad to the target size

    if dim == 3:
        cropped = np.pad(
            cropped,
            (
                (-min(0, crop_x_lb), max(crop_x_ub - full_size[0], 0)),
                (-min(0, crop_y_lb), max(crop_y_ub - full_size[1], 0)),
                (-min(0, crop_z_lb), max(crop_z_ub - full_size[2], 0)),
            ),
            mode=pad_mode,
            **pad_kwargs_data,
        )
    else:
        cropped = np.pad(
            cropped,
            (
                (-min(0, crop_x_lb), max(crop_x_ub - full_size[0], 0)),
                (-min(0, crop_y_lb), max(crop_y_ub - full_size[1], 0)),
            ),
            mode=pad_mode,
            **pad_kwargs_data,
        )

    cropped = np.expand_dims(cropped, axis=0)

    return cropped


def find_bbox_from_mask(mask):
    """_summary_

    _extended_summary_

    Args:
        label_mask (_type_): _description_

    Returns:
        Describe the type and semantics of the return value. If the function only returns None, this section is not required.
        If exists multiple return values with individual names, describe such a return value as: “Returns: A tuple (mat_a, mat_b), where mat_a is …, and …”.
    """
    dim = len(mask.shape)

    label_mask = mask.copy()
    nonzero_regions = np.where(label_mask > 0)
    x_min, x_max = nonzero_regions[0].min(), nonzero_regions[0].max()
    y_min, y_max = nonzero_regions[1].min(), nonzero_regions[1].max()
    if dim == 3:
        z_min, z_max = nonzero_regions[2].min(), nonzero_regions[2].max()
        h, w, d = x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1
        return (x_min, y_min, z_min, h, w, d)
    else:
        h, w = x_max - x_min + 1, y_max - y_min + 1
        return (x_min, y_min, h, w)


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def subfiles(
    folder: str,
    join: bool = True,
    prefix: str = None,
    suffix: str = None,
    sort: bool = True,
) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def recursive_find_python_class(folder, trainer_name, current_module):
    """
    To find python class recursively in given folder
    """
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    [os.path.join(folder[0], modname)],
                    trainer_name,
                    current_module=next_current_module,
                )
            if tr is not None:
                break

    return tr


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if not seq:
        return ""
    if len(seq) == 1:
        return f"'{seq[0]}'"

    head = "'" + "', '".join([str(item) for item in seq[:-1]]) + "'"
    tail = (
        f"{'' if separate_last and len(seq) == 2 else ','} {separate_last}'{seq[-1]}'"
    )

    return head + tail


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
