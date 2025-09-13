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
from rtnet.data.data_loader.cfg_base import Cfg

cfg = Cfg()
from rtnet.data.data_loader.data_image_utils.utils import recursive_find_python_class


def load_pickle(file: str, mode: str = "rb"):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


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


# def get_default_configuration(res, task, data_loader, plans_identifier=cfg.default_plans_identifier,
#                               search_in=(rtnet.__path__[0], "data", "data_loader", "data_image_dataloading"),
#                               base_module='rtnet.data.data_loader.data_image_dataloading'):
def get_default_configuration(
    res,
    task,
    data_loader,
    plans_identifier=cfg.default_plans_identifier,
    search_in=(rtnet.__path__[0], "data", "datasets"),
    base_module="rtnet.data.datasets",
):
    assert res in [
        "2d",
        "3d_lowres",
        "3d_fullres",
        "3d_cascade_fullres",
    ], "network can only be one of the following: '3d_lowres', '3d_fullres', '3d_cascade_fullres'"

    dataset_directory = os.path.join(cfg.preprocessing_output_dir, task)

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
            cfg.preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl"
        )
        plans_file_json = os.path.join(
            cfg.preprocessing_output_dir, task, plans_identifier + "_plans_2D.json"
        )
    else:
        plans_file_pkl = os.path.join(
            cfg.preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl"
        )
        plans_file_json = os.path.join(
            cfg.preprocessing_output_dir, task, plans_identifier + "_plans_3D.json"
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
        cfg.network_training_output_dir,
        res,
        task,
        data_loader + "__" + plans_identifier,
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
    return plans_file, output_folder_name, dataset_directory, stage, data_loader_class
