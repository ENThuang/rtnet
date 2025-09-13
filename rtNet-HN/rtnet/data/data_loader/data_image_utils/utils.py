#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep. 18 2022

@author: Dazhou Guo
"""

import json
import os
import pickle
import shutil
import importlib
import pkgutil
import numpy as np
from os.path import join as join
from copy import deepcopy
import SimpleITK as sitk
from collections import OrderedDict
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isdir,
    maybe_mkdir_p,
    subfiles,
    subdirs,
    isfile,
)
from rtnet.data.data_loader.cfg_base import Cfg

cfg = Cfg()
# default_num_threads, preprocessing_output_dir, rtNet_raw_data, rtNet_cropped_data, network_training_output_dir
from rtnet.data.data_loader.data_image_planning.DatasetAnalyzer import DatasetAnalyzer
from rtnet.data.data_loader.data_image_preprocessing.cropping import ImageCropper


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
                    [join(folder[0], modname)],
                    trainer_name,
                    current_module=next_current_module,
                )
            if tr is not None:
                break

    return tr


def convert_id_to_task_name(task_id: int):
    startswith = "Task%03.0d" % task_id
    if cfg.preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(
            cfg.preprocessing_output_dir, prefix=startswith, join=False
        )
    else:
        candidates_preprocessed = []

    if cfg.rtNet_raw_data is not None:
        candidates_raw = subdirs(cfg.rtNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if cfg.rtNet_cropped_data is not None:
        candidates_cropped = subdirs(
            cfg.rtNet_cropped_data, prefix=startswith, join=False
        )
    else:
        candidates_cropped = []

    candidates_trained_models = []
    if cfg.network_training_output_dir is not None:
        for m in ["2d", "3d_lowres", "3d_fullres"]:
            if isdir(join(cfg.network_training_output_dir, m)):
                candidates_trained_models += subdirs(
                    join(cfg.network_training_output_dir, m),
                    prefix=startswith,
                    join=False,
                )

    all_candidates = (
        candidates_cropped
        + candidates_preprocessed
        + candidates_raw
        + candidates_trained_models
    )
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError(
            "More than one task name found for task id %d. Please correct that. (I looked in the "
            "following folders:\n%s\n%s\n%s"
            % (
                task_id,
                cfg.rtNet_raw_data,
                cfg.preprocessing_output_dir,
                cfg.rtNet_cropped_data,
            )
        )
    if len(unique_candidates) == 0:
        raise RuntimeError(
            "Could not find a task with the ID %d. Make sure the requested task ID exists and that "
            "rtNet knows where raw and preprocessed data are located (see Documentation - "
            "Installation). Here are your currently defined folders:\nrtNet_preprocessed=%s\nRESULTS_"
            "FOLDER=%s\nrtNet_raw_data_base=%s\nIf something is not right, adapt your environemnt "
            "variables."
            % (
                task_id,
                os.environ.get("rtNet_preprocessed")
                if os.environ.get("rtNet_preprocessed") is not None
                else "None",
                os.environ.get("RESULTS_FOLDER")
                if os.environ.get("RESULTS_FOLDER") is not None
                else "None",
                os.environ.get("rtNet_raw_data_base")
                if os.environ.get("rtNet_raw_data_base") is not None
                else "None",
            )
        )
    return unique_candidates[0]


def split_4d_nifti(filename, output_folder):
    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = filename.split("/")[-1]
    if dim == 3:
        shutil.copy(filename, join(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError(
            "Unexpected dimensionality: %d of file %s, cannot split" % (dim, filename)
        )
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4, 4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(
                img_itk_new, join(output_folder, file_base[:-7] + "_%04.0d.nii.gz" % i)
            )


def get_pool_and_conv_props_poolLateV2(
    patch_size, min_feature_map_size, max_numpool, spacing
):
    """

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    initial_spacing = deepcopy(spacing)
    reach = max(initial_spacing)
    dim = len(patch_size)

    num_pool_per_axis = get_network_numpool(
        patch_size, max_numpool, min_feature_map_size
    )

    net_num_pool_op_kernel_sizes = []
    net_conv_kernel_sizes = []
    net_numpool = max(num_pool_per_axis)

    current_spacing = spacing
    for p in range(net_numpool):
        reached = [current_spacing[i] / reach > 0.5 for i in range(dim)]
        pool = [2 if num_pool_per_axis[i] + p >= net_numpool else 1 for i in range(dim)]
        if all(reached):
            conv = [3] * dim
        else:
            conv = [3 if not reached[i] else 1 for i in range(dim)]
        net_num_pool_op_kernel_sizes.append(pool)
        net_conv_kernel_sizes.append(conv)
        current_spacing = [i * j for i, j in zip(current_spacing, pool)]

    net_conv_kernel_sizes.append([3] * dim)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    return (
        num_pool_per_axis,
        net_num_pool_op_kernel_sizes,
        net_conv_kernel_sizes,
        patch_size,
        must_be_divisible_by,
    )


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim

    while True:
        # This is a problem because sometimes we have spacing 20, 50, 50 and we want to still keep pooling.
        # Here we would stop however. This is not what we want! Fixed in get_pool_and_conv_propsv2
        min_spacing = min(current_spacing)
        valid_axes_for_pool = [
            i for i in range(dim) if current_spacing[i] / min_spacing < 2
        ]
        axes = []
        for a in range(dim):
            my_spacing = current_spacing[a]
            partners = [
                i
                for i in range(dim)
                if current_spacing[i] / my_spacing < 2
                and my_spacing / current_spacing[i] < 2
            ]
            if len(partners) > len(axes):
                axes = partners
        conv_kernel_size = [3 if i in axes else 1 for i in range(dim)]

        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        # before = len(valid_axes_for_pool)
        valid_axes_for_pool = [
            i
            for i in valid_axes_for_pool
            if current_size[i] >= 2 * min_feature_map_size
        ]
        # after = len(valid_axes_for_pool)
        # if after == 1 and before > 1:
        #    break

        valid_axes_for_pool = [
            i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool
        ]

        if len(valid_axes_for_pool) == 0:
            break

        # print(current_spacing, current_size)

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(conv_kernel_size)
        # print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3] * dim)
    return (
        num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        patch_size,
        must_be_divisible_by,
    )


def get_pool_and_conv_props_v2(spacing, patch_size, min_feature_map_size, max_numpool):
    """

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        valid_axes_for_pool = [
            i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size
        ]
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]

        # find axis that are within factor of 2 within smallest spacing
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [
            i
            for i in valid_axes_for_pool
            if current_spacing[i] / min_spacing_of_valid < 2
        ]

        # max_numpool constraint
        valid_axes_for_pool = [
            i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool
        ]

        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break

        # now we need to find kernel sizes
        # kernel sizes are initialized to 1. They are successively set to 3 when their associated axis becomes within
        # factor 2 of min_spacing. Once they are 3 they remain 3
        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            else:
                if spacings_of_axes[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))
        # print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3] * dim)
    return (
        num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        patch_size,
        must_be_divisible_by,
    )


def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisibly by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [
        shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i]
        for i in range(len(shape))
    ]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp


def get_network_numpool(patch_size, maxpool_cap=999, min_feature_map_size=4):
    network_numpool_per_axis = np.floor(
        [np.log(i / min_feature_map_size) / np.log(2) for i in patch_size]
    ).astype(int)
    network_numpool_per_axis = [min(i, maxpool_cap) for i in network_numpool_per_axis]
    return network_numpool_per_axis


def split_4d(
    input_folder, num_processes=cfg.default_num_threads, overwrite_task_output_id=None
):
    assert (
        isdir(join(input_folder, "imagesTr"))
        and isdir(join(input_folder, "labelsTr"))
        and isfile(join(input_folder, "dataset.json"))
    ), (
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the "
        "imagesTr and labelsTr subfolders and the dataset.json file"
    )

    while input_folder.endswith("/"):
        input_folder = input_folder[:-1]

    full_task_name = input_folder.split("/")[-1]

    assert full_task_name.startswith(
        "Task"
    ), "The input folder must point to a folder that starts with TaskXX_"

    first_underscore = full_task_name.find("_")
    assert (
        first_underscore == 6
    ), "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"

    input_task_id = int(full_task_name[4:6])
    if overwrite_task_output_id is None:
        overwrite_task_output_id = input_task_id

    task_name = full_task_name[7:]

    output_folder = join(
        cfg.rtNet_raw_data, "Task%03.0d_" % overwrite_task_output_id + task_name
    )

    if isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    maybe_mkdir_p(output_folder)
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = join(output_folder, subdir)
        if not isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join(input_folder, subdir)
        nii_files = [
            join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")
        ]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(join(input_folder, "labelsTr"), join(output_folder, "labelsTr"))

    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(join(input_folder, "dataset.json"), output_folder)


def create_lists_from_splitted_dataset(base_folder_splitted, split="training"):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d[split]
    num_modalities = len(d["modality"].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(
                join(
                    base_folder_splitted,
                    tr["image"].replace("./", "")[:-7] + "_%04.0d.nii.gz" % mod,
                )
            )
            # cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
            #                     "_%04.0d.nii.gz" % mod))
        # cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1])) # removed for cls tasks
        lists.append(cur_pat)
    return lists, {int(i): d["modality"][str(i)] for i in d["modality"].keys()}


def create_lists_from_splitted_dataset_folder(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    caseIDs = get_caseIDs_from_splitted_dataset_folder(folder)
    list_of_lists = []
    for f in caseIDs:
        list_of_lists.append(
            subfiles(folder, prefix=f, suffix=".nii.gz", join=True, sort=True)
        )
    return list_of_lists


def get_caseIDs_from_splitted_dataset_folder(folder):
    files = subfiles(folder, suffix=".nii.gz", join=False)
    # all files must be .nii.gz and have 4 digit modality index
    files = [i[:-12] for i in files]
    # only unique patient ids
    files = np.unique(files)
    return files


def crop(task_string, override=False, num_threads=cfg.default_num_threads):
    cropped_out_dir = join(cfg.rtNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(cfg.rtNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join(cfg.rtNet_raw_data, task_string, "dataset.json"), cropped_out_dir)


def analyze_dataset(
    task_string,
    override=False,
    collect_intensityproperties=True,
    num_processes=cfg.default_num_threads,
):
    cropped_out_dir = join(cfg.rtNet_cropped_data, task_string)
    dataset_analyzer = DatasetAnalyzer(
        cropped_out_dir, overwrite=override, num_processes=num_processes
    )
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)


def plan_and_preprocess(
    task_string,
    processes_lowres=cfg.default_num_threads,
    processes_fullres=3,
    no_preprocessing=False,
):
    from rtnet.data.data_loader.data_image_planning.experiment_planner_2D import (
        ExperimentPlanner2D,
    )
    from rtnet.data.data_loader.data_image_planning.experiment_planner_base import (
        ExperimentPlanner_Base,
    )

    preprocessing_output_dir_this_task_train = join(
        cfg.preprocessing_output_dir, task_string
    )
    cropped_out_dir = join(cfg.rtNet_cropped_data, task_string)
    maybe_mkdir_p(preprocessing_output_dir_this_task_train)

    shutil.copy(
        join(cropped_out_dir, "dataset_properties.pkl"),
        preprocessing_output_dir_this_task_train,
    )
    shutil.copy(
        join(cfg.rtNet_raw_data, task_string, "dataset.json"),
        preprocessing_output_dir_this_task_train,
    )

    exp_planner = ExperimentPlanner_Base(
        cropped_out_dir, preprocessing_output_dir_this_task_train
    )
    exp_planner.plan_experiment()
    if not no_preprocessing:
        exp_planner.run_preprocessing((processes_lowres, processes_fullres))

    exp_planner = ExperimentPlanner2D(
        cropped_out_dir, preprocessing_output_dir_this_task_train
    )
    exp_planner.plan_experiment()
    if not no_preprocessing:
        exp_planner.run_preprocessing(processes_fullres)

    # write which class is in which slice to all training cases (required to speed up 2D Dataloader)
    # This is done for all data so that if we wanted to use them with 2D we could do so

    if not no_preprocessing:
        p = Pool(cfg.default_num_threads)

        # if there is more than one my_data_identifier (different brnaches) then this code will run for all of them if
        # they start with the same string. not problematic, but not pretty
        stages = [
            i
            for i in subdirs(
                preprocessing_output_dir_this_task_train, join=True, sort=True
            )
            if i.split("/")[-1].find("stage") != -1
        ]
        for s in stages:
            print(s.split("/")[-1])
            list_of_npz_files = subfiles(s, True, None, ".npz", True)
            list_of_pkl_files = [i[:-4] + ".pkl" for i in list_of_npz_files]
            all_classes = []
            for pk in list_of_pkl_files:
                with open(pk, "rb") as f:
                    props = pickle.load(f)
                all_classes_tmp = np.array(props["classes"])
                all_classes.append(all_classes_tmp[all_classes_tmp >= 0])
            p.map(
                add_classes_in_slice_info,
                zip(list_of_npz_files, list_of_pkl_files, all_classes),
            )
        p.close()
        p.join()


def add_classes_in_slice_info(args):
    """
    We need this for 2D dataloader with oversampling. As of now it will detect slices that contain specific classes
    at run time, meaning it needs to iterate over an entire patient just to extract one slice. That is obviously bad,
    so we are doing this once beforehand and just give the dataloader the info it needs in the patients pkl file.

    """
    npz_file, pkl_file, all_classes = args
    seg_map = np.load(npz_file)["data"][-1]
    with open(pkl_file, "rb") as f:
        props = pickle.load(f)
    # if props.get('classes_in_slice_per_axis') is not None:
    print(pkl_file)
    # this will be a dict of dict where the first dict encodes the axis along which a slice is extracted in its keys.
    # The second dict (value of first dict) will have all classes as key and as values a list of all slice ids that
    # contain this class
    classes_in_slice = OrderedDict()
    for axis in range(3):
        other_axes = tuple([i for i in range(3) if i != axis])
        classes_in_slice[axis] = OrderedDict()
        for c in all_classes:
            valid_slices = np.where(np.sum(seg_map == c, axis=other_axes) > 0)[0]
            classes_in_slice[axis][c] = valid_slices

    number_of_voxels_per_class = OrderedDict()
    for c in all_classes:
        number_of_voxels_per_class[c] = np.sum(seg_map == c)

    props["classes_in_slice_per_axis"] = classes_in_slice
    props["number_of_voxels_per_class"] = number_of_voxels_per_class

    with open(pkl_file, "wb") as f:
        pickle.dump(props, f)
