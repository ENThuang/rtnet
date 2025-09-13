#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep. 18 2022

@author: Dazhou Guo
"""

import os
import rtnet
import pickle
import json
import numpy as np
from multiprocessing import Pool
from collections import OrderedDict
from sklearn.model_selection import KFold
from typing import List

from rtnet.utils import subfiles, recursive_find_python_class


def determine_normalization_scheme(dataset_properties):
    schemes = OrderedDict()
    modalities = dataset_properties["modalities"]
    num_modalities = len(list(modalities.keys()))

    for i in range(num_modalities):
        if modalities[i] == "CT" or modalities[i] == "ct":
            schemes[i] = "CT"
        elif modalities[i] == "noNorm":
            schemes[i] = "noNorm"
        elif "support" in modalities[i].lower():
            schemes[i] = "support"
        else:
            schemes[i] = "nonCT"

    return schemes


def load_pickle(file: str, mode: str = "rb"):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = "wb") -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file, "r") as f:
        a = json.load(f)
    return a


def summarize_plans(plans):
    print("num_classes: ", plans["num_classes"])
    print("modalities: ", plans["modalities"])
    print("use_mask_for_norm", plans["use_mask_for_norm"])
    # print("keep_only_largest_region", plans['keep_only_largest_region'])
    # print("min_region_size_per_class", plans['min_region_size_per_class'])
    # print("min_size_per_class", plans['min_size_per_class'])
    print("normalization_schemes", plans["normalization_schemes"])
    print("stages...\n")

    for i in list(plans["plans_per_stage"].keys()):
        print("stage: ", i)
        print(plans["plans_per_stage"][i])
        print("")


def subdirs(
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
        if os.path.isdir(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def convert_id_to_task_name(
    task_id: int,
    preprocessing_output_dir,
    rtNet_raw_data,
    rtNet_cropped_data,
    network_training_output_dir,
):
    startswith = "Task%03.0d" % task_id
    if preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(
            preprocessing_output_dir, prefix=startswith, join=False
        )
    else:
        candidates_preprocessed = []

    if rtNet_raw_data is not None:
        candidates_raw = subdirs(rtNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if rtNet_cropped_data is not None:
        candidates_cropped = subdirs(rtNet_cropped_data, prefix=startswith, join=False)
    else:
        candidates_cropped = []

    candidates_trained_models = []
    if network_training_output_dir is not None:
        for m in ["2d", "3d_lowres", "3d_fullres"]:
            if os.path.isdir(os.path.join(network_training_output_dir, m)):
                candidates_trained_models += subdirs(
                    os.path.join(network_training_output_dir, m),
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
            % (task_id, rtNet_raw_data, preprocessing_output_dir, rtNet_cropped_data)
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


def get_default_configuration(
    res,
    task,
    data_loader,
    plans_identifier,
    preprocessing_output_dir,
    rtNet_raw_data,
    rtNet_cropped_data,
    network_training_output_dir,
    search_in=(rtnet.__path__[0], "data_loader", "data_image_dataloading"),
    base_module="rtnet.data_loader.data_image_dataloading",
):
    assert res in [
        "2d",
        "3d_lowres",
        "3d_fullres",
        "3d_cascade_fullres",
    ], "network can only be one of the following: '3d_lowres', '3d_fullres', '3d_cascade_fullres'"
    if isinstance(task, List):
        if isinstance(int(task[0]), int):
            task = convert_id_to_task_name(
                int(task[0]),
                preprocessing_output_dir,
                rtNet_raw_data,
                rtNet_cropped_data,
                network_training_output_dir,
            )
        else:
            task = task[0]
    elif isinstance(task, str):
        task = convert_id_to_task_name(
            int(task),
            preprocessing_output_dir,
            rtNet_raw_data,
            rtNet_cropped_data,
            network_training_output_dir,
        )
    else:
        raise ValueError("task must be a string or a list of strings or integers")

    dataset_directory = os.path.join(preprocessing_output_dir, task)

    def convert_keys_to_int(d: dict):
        new_dict = {}
        for k, v in d.items():
            try:
                new_key = int(k)
            except ValueError:
                new_key = k
            if type(v) == dict:
                v = convert_keys_to_int(v)
            new_dict[new_key] = v
        return new_dict

    if res == "2d":
        plans_file_pkl = os.path.join(
            preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl"
        )
        plans_file_json = os.path.join(
            preprocessing_output_dir, task, plans_identifier + "_plans_2D.json"
        )
    else:
        plans_file_pkl = os.path.join(
            preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl"
        )
        plans_file_json = os.path.join(
            preprocessing_output_dir, task, plans_identifier + "_plans_3D.json"
        )

    try:
        plans = load_pickle(plans_file_pkl)
    except FileNotFoundError:
        plans = convert_keys_to_int(load_json(plans_file_json))
        plans_file = plans_file_json
    else:
        plans_file = plans_file_pkl
    possible_stages = list(plans["plans_per_stage"].keys())

    if (res == "3d_cascade_fullres" or res == "3d_lowres") and len(
        possible_stages
    ) == 1:
        raise RuntimeError(
            "3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
            "not require the cascade. Run 3d_fullres instead"
        )

    if res == "2d" or res == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    data_loader_class = recursive_find_python_class(
        [os.path.join(*search_in)], data_loader, current_module=base_module
    )

    output_folder_name = os.path.join(
        network_training_output_dir, res, task, data_loader + "__" + plans_identifier
    )

    print("###############################################")
    print("Loading data using: %s" % res)
    print("My data loader class is: ", data_loader_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans)
    print("I am using stage %s from these plans" % str(stage))
    print(
        "\nI am using data from this folder: ",
        os.path.join(dataset_directory, plans["data_identifier"]),
    )
    print("###############################################")
    return (
        plans,
        plans_file,
        output_folder_name,
        dataset_directory,
        stage,
        data_loader_class,
    )


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def unpack_dataset(folder, threads, key="data"):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key] * len(npz_files)))
    p.close()
    p.join()


def get_case_identifiers(folder):
    case_identifiers = [
        i[:-4]
        for i in os.listdir(folder)
        if i.endswith("npz") and (i.find("segFromPrevStage") == -1)
    ]
    return case_identifiers


def load_dataset(
    folder, num_cases_properties_loading_threshold=1000, recist_lo=None, recist_hi=None
):
    # we don't load the actual data but instead return the filename to the np file.
    print("loading dataset")

    # load pre-calculated RECIST to filter out easy cases
    ins_info = {}
    # pos_cnt, neg_cnt = 0, 0
    processed_folder = os.path.dirname(folder)
    if recist_lo is not None and recist_hi is not None:
        import csv

        with open(os.path.join(processed_folder, "cropping_list.csv"), "r") as data:
            for line in csv.DictReader(data):
                s_pth = line["path"].replace("/", "_")
                s_basename = line["basename"]
                filename = s_pth + "_" + s_basename if s_pth else s_basename
                filename += "_pos" if int(line["label"]) == 1 else "_neg"
                ins_info[filename] = {"recist": line["recist"], "label": line["label"]}

    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        # first check if need to filter out easy cases
        if len(ins_info) > 0:
            assert c in ins_info, (
                "Case %s is not in the instance info dict ins_info" % c
            )
            if int(ins_info[c]["label"]) == 0:
                if (
                    recist_lo["neg"] is not None
                    and float(ins_info[c]["recist"]) <= recist_lo["neg"]
                ):
                    continue
            else:
                if (
                    recist_hi["pos"] is not None
                    and float(ins_info[c]["recist"]) >= recist_hi["pos"]
                ):
                    continue

        # if 'pos' in c:
        #     pos_cnt += 1
        # else:
        #     neg_cnt += 1

        dataset[c] = OrderedDict()
        dataset[c]["data_file"] = os.path.join(folder, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]["properties_file"] = os.path.join(folder, "%s.pkl" % c)

        if dataset[c].get("seg_from_prev_stage_file") is not None:
            dataset[c]["seg_from_prev_stage_file"] = os.path.join(
                folder, "%s_segs.npz" % c
            )

    # print("Sample distribution: positive {}, negative {}".format(pos_cnt, neg_cnt))

    # if len(case_identifiers) <= num_cases_properties_loading_threshold:
    #     print('loading all case properties')
    #     for i in dataset.keys():
    #         dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])
    # else:
    #     raise "Exceeded the maxium case number %d, please check the argument \
    #         'num_cases_properties_loading_threshold'"

    print("loading all case properties")
    if len(dataset) > num_cases_properties_loading_threshold:
        print(
            "Warning: total case number {} exceeds the maximum loading threashold {}".format(
                len(dataset), num_cases_properties_loading_threshold
            )
        )
    for i in dataset.keys():
        dataset[i]["properties"] = load_pickle(dataset[i]["properties_file"])
    return dataset


def do_split(fold, dataset, dataset_directory):
    """
    The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
    so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
    Sometimes you may want to create your own split for various reasons. For this you will need to create your own
    splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
    it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
    and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
    use a random 80:20 data split.
    :return:
    """
    if fold == "all":
        # if fold==all then we use all images for training and validation
        tr_keys = val_keys = list(dataset.keys())
    else:
        splits_file = os.path.join(dataset_directory, "splits_final.pkl")
        # splits_file = os.path.join(dataset_directory, "splits_final_allpath.pkl")

        # if the split file does not exist we need to create it
        if not os.path.isfile(splits_file):
            # print("Creating new 5-fold cross-validation split...")
            print("Creating new 4-fold cross-validation split...")
            splits = []
            all_keys_sorted = np.sort(list(dataset.keys()))
            # kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            kfold = KFold(n_splits=4, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]["train"] = train_keys
                splits[-1]["val"] = test_keys
            write_pickle(splits, splits_file)

        else:
            print("Using splits from existing split file:", splits_file)
            splits = load_pickle(splits_file)
            print("The split file contains %d splits." % len(splits))

        print("Desired fold for training: %d" % fold)
        if fold < len(splits):
            tr_keys = splits[fold]["train"]
            val_keys = splits[fold]["val"]
            # print("This split has %d training and %d validation cases." % (len(tr_keys), len(val_keys)))
        else:
            print(
                "INFO: You requested fold %d for training but splits "
                "contain only %d folds. I am now creating a "
                "random (but seeded) 80:20 split!" % (fold, len(splits))
            )
            # if we request a fold that is not in the split file, create a random 80:20 split
            rnd = np.random.RandomState(seed=12345 + fold)
            keys = np.sort(list(dataset.keys()))
            idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
            idx_val = [i for i in range(len(keys)) if i not in idx_tr]
            tr_keys = [keys[i] for i in idx_tr]
            val_keys = [keys[i] for i in idx_val]
            print(
                "This random 80:20 split has %d training and %d validation cases."
                % (len(tr_keys), len(val_keys))
            )

    tr_keys.sort()
    val_keys.sort()
    pos_cnt, neg_cnt = 0, 0

    dataset_tr = OrderedDict()
    for i in tr_keys:
        if i not in dataset:
            continue
        pos_cnt += 1 if "pos" in i else 0
        neg_cnt += 1 if "neg" in i else 0
        dataset_tr[i] = dataset[i]

    dataset_val = OrderedDict()
    for i in val_keys:
        if i not in dataset:
            continue
        pos_cnt += 1 if "pos" in i else 0
        neg_cnt += 1 if "neg" in i else 0
        dataset_val[i] = dataset[i]

    print(
        "This split has %d training and %d validation cases."
        % (len(dataset_tr), len(dataset_val))
    )
    print("Sample distribution: positive {}, negative {}".format(pos_cnt, neg_cnt))

    return dataset_tr, dataset_val


def unpack_npz_and_get_train_split(
    fold,
    threads,
    dataset_directory,
    plans,
    stage,
    recist_lo=None,
    recist_hi=None,
    in_use_keys=None,
):
    from rtnet.utils import wait_for_the_master

    # path to the processed imaging data
    folder_with_preprocessed_data = os.path.join(
        dataset_directory, plans["data_identifier"] + "_stage%d" % stage
    )

    # unpack npz files to npy for faster data loading
    with wait_for_the_master():
        print("unpacking dataset")
        unpack_dataset(folder_with_preprocessed_data, threads)
        print("done")

    # list npz files in the directory
    dataset = load_dataset(
        folder_with_preprocessed_data, recist_lo=recist_lo, recist_hi=recist_hi
    )
    # remove samples with specified keys
    if in_use_keys != "all" and (
        isinstance(in_use_keys, List) and len(in_use_keys) > 0
    ):
        dataset = remove_samples_not_used(dataset, in_use_keys)
    # generate 5-fold cross-validation split
    dataset_tr, dataset_val = do_split(fold, dataset, dataset_directory)

    return dataset_tr, dataset_val


def remove_samples_not_used(dataset, keys):
    if not isinstance(keys, set):
        keys = set(keys)
    out = OrderedDict()
    for k, v in dataset.items():
        if k.split("_")[1] not in keys:
            continue
        out[k] = v
    print("Keep {} cases with keys {}".format(len(dataset) - len(out), list(keys)))
    return out
