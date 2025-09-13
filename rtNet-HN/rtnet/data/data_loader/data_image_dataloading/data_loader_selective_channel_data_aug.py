#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep. 18 2022

@author: Dazhou Guo
"""

from collections import OrderedDict
from time import time, sleep
from datetime import datetime
import numpy as np
import sys
from rtnet.data.data_loader.data_image_data_aug.data_augmentation_SelectiveChannelDA import (
    get_selective_channel_augmentation,
)
from rtnet.data.data_loader.data_image_data_aug.default_data_augmentation import (
    default_2D_augmentation_params,
    get_patch_size,
    default_3D_augmentation_params,
)
from rtnet.data.data_loader.data_image_dataloading.dataset_loading import (
    unpack_dataset,
    load_dataset,
    DataLoader3D,
    DataLoader2D,
)
from sklearn.model_selection import KFold
from batchgenerators.utilities.file_and_folder_operations import *
from rtnet.data.data_loader.cfg_base import Cfg

cfg = Cfg()


class rtNetDataLoader_SelectiveChannelDA(object):
    def __init__(
        self,
        plans_file,
        patch_size,
        fold,
        output_folder=None,
        dataset_directory=None,
        stage=None,
        unpack_data=True,
        deterministic=True,
        net_num_pool_op_kernel_sizes=[1, 1, 1],
        fp16=False,
    ):
        self.plans_file = plans_file
        self.deep_supervision_scales = None
        self.pin_memory = True
        self.unpack_data = unpack_data
        self.was_initialized = False
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.plans = None
        self.log_file = None
        self.stage = stage
        self.fold = fold
        self.fp16 = fp16
        self.dl_tr = self.dl_val = None
        self.tr_gen = self.val_gen = None
        self.deterministic = deterministic
        self.oversample_foreground_percent = 0.33
        self.pad_all_sides = None
        self.patch_size = patch_size
        self.net_num_pool_op_kernel_sizes = net_num_pool_op_kernel_sizes

        self.num_input_channels = (
            self.num_classes
        ) = (
            self.net_pool_per_axis
        ) = (
            self.batch_size
        ) = (
            self.threeD
        ) = (
            self.base_num_features
        ) = (
            self.intensity_properties
        ) = (
            self.normalization_schemes
        ) = (
            self.net_conv_kernel_sizes
        ) = self.folder_with_preprocessed_data = self.do_dummy_2D_aug = None

        self.classes = (
            self.do_dummy_2D_aug
        ) = (
            self.use_mask_for_norm
        ) = (
            self.only_keep_largest_connected_component
        ) = self.min_region_size_per_class = self.min_size_per_class = None

        self.basic_generator_patch_size = (
            self.data_aug_params
        ) = self.transpose_forward = self.transpose_backward = None
        self.dataset = None
        self.conv_per_stage = None

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)
            self.normalization_schemes = self.plans["normalization_schemes"]
            self.setup_DA_params()
            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )

                self.tr_gen, self.val_gen = get_selective_channel_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                    also_print_to_console=False,
                )
            else:
                pass
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """

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

        format = self.plans_file.split(".")[-1]
        if format == "pkl":
            self.plans = load_pickle(self.plans_file)
        elif format == "json":
            with open(self.plans_file, "r") as f:
                self.plans = convert_keys_to_int(json.load(f))
        else:
            raise RuntimeError(
                "Loading plan file format {} not supported!".format(format)
            )

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans["plans_per_stage"].keys())) == 1, (
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the "
                "case. Please specify which stage of the cascade must be trained"
            )
            self.stage = list(plans["plans_per_stage"].keys())[0]
        self.plans = plans

        stage_plans = self.plans["plans_per_stage"][self.stage]
        self.batch_size = cfg.data_loader_planning_and_preprocessing_batch_size
        if self.batch_size > 1:
            if self.patch_size is None:
                raise RuntimeError("You must setup a patch size for cropping.")
        else:
            self.batch_size = 1
        self.do_dummy_2D_aug = cfg.do_dummy_2D_data_aug
        self.intensity_properties = plans["dataset_properties"]["intensityproperties"]
        self.normalization_schemes = plans["normalization_schemes"]
        self.num_input_channels = plans["num_modalities"]
        self.num_classes = (
            plans["num_classes"] + 1
        )  # background is no longer in num_classes
        self.classes = plans["all_classes"]
        self.use_mask_for_norm = plans["use_mask_for_norm"]
        self.only_keep_largest_connected_component = plans["keep_only_largest_region"]
        self.min_region_size_per_class = plans["min_region_size_per_class"]
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if (
            plans.get("transpose_forward") is None
            or plans.get("transpose_backward") is None
        ):
            print(
                "WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!"
            )
            plans["transpose_forward"] = [0, 1, 2]
            plans["transpose_backward"] = [0, 1, 2]
        self.transpose_forward = plans["transpose_forward"]
        self.transpose_backward = plans["transpose_backward"]

        if self.patch_size is not None and len(self.patch_size) == 2:
            self.threeD = False
        elif self.patch_size is None or len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError(
                "invalid patch size in plans file: %s" % str(self.patch_size)
            )

        if (
            "conv_per_stage" in plans.keys()
        ):  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans["conv_per_stage"]
        else:
            self.conv_per_stage = 2

    def do_split(self):
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
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]["train"] = train_keys
                    splits[-1]["val"] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file(
                    "Using splits from existing split file:", splits_file
                )
                splits = load_pickle(splits_file)
                self.print_to_log_file(
                    "The split file contains %d splits." % len(splits)
                )

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]["train"]
                val_keys = splits[self.fold]["val"]
                self.print_to_log_file(
                    "This split has %d training and %d validation cases."
                    % (len(tr_keys), len(val_keys))
                )
            else:
                self.print_to_log_file(
                    "INFO: You requested fold %d for training but splits "
                    "contain only %d folds. I am now creating a "
                    "random (but seeded) 80:20 split!" % (self.fold, len(splits))
                )
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file(
                    "This random 80:20 split has %d training and %d validation cases."
                    % (len(tr_keys), len(val_keys))
                )

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """
        self.deep_supervision_scales = [[1, 1, 1]] + list(
            list(i)
            for i in 1
            / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
        )[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params["rotation_x"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_y"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_z"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params[
                    "elastic_deform_alpha"
                ] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params[
                    "elastic_deform_sigma"
                ] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params[
                    "rotation_x"
                ]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (
                    -15.0 / 360 * 2.0 * np.pi,
                    15.0 / 360 * 2.0 * np.pi,
                )
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.patch_size is not None:
            if self.do_dummy_2D_aug:
                self.basic_generator_patch_size = get_patch_size(
                    self.patch_size[1:],
                    self.data_aug_params["rotation_x"],
                    self.data_aug_params["rotation_y"],
                    self.data_aug_params["rotation_z"],
                    self.data_aug_params["scale_range"],
                )
                self.basic_generator_patch_size = np.array(
                    [self.patch_size[0]] + list(self.basic_generator_patch_size)
                )

            else:
                self.basic_generator_patch_size = get_patch_size(
                    self.patch_size,
                    self.data_aug_params["rotation_x"],
                    self.data_aug_params["rotation_y"],
                    self.data_aug_params["rotation_z"],
                    self.data_aug_params["scale_range"],
                )
        else:
            self.basic_generator_patch_size = None

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["selected_seg_channels"] = [0]
        self.data_aug_params["patch_size_for_spatialtransform"] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2
        self.data_aug_params["normalization_schemes"] = self.normalization_schemes

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
            dl_val = DataLoader3D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        else:
            dl_tr = DataLoader2D(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
            dl_val = DataLoader2D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        return dl_tr, dl_val

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(
                self.output_folder,
                "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt"
                % (
                    timestamp.year,
                    timestamp.month,
                    timestamp.day,
                    timestamp.hour,
                    timestamp.minute,
                    timestamp.second,
                ),
            )
            with open(self.log_file, "w") as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, "a+") as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print(
                    "%s: failed to log: " % datetime.fromtimestamp(timestamp),
                    sys.exc_info(),
                )
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)


if __name__ == "__main__":
    from rtnet.data.data_loader.data_image_utils.utils_data_loading import *
    from rtnet.data.data_loader.data_image_utils.utils import convert_id_to_task_name
    from rtnet.data.data_loader.cfg_base import Cfg

    cfg = Cfg()
    preprocessing_output_dir = cfg.preprocessing_output_dir
    # import os
    # import json

    res = cfg.res
    task = int(cfg.task_ids[0])
    task = convert_id_to_task_name(task)
    data_loader = cfg.default_data_loader
    plans_identifier = cfg.default_plans_identifier

    (
        plans_file,
        output_folder_name,
        dataset_directory,
        stage,
        data_loader_class,
    ) = get_default_configuration(res, task, data_loader, plans_identifier)
    data_loader = data_loader_class(
        plans_file,
        cfg.input_patch_size,
        0,
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        stage=stage,
        unpack_data=True,
        deterministic=True,
        net_num_pool_op_kernel_sizes=cfg.data_loader_planning_and_preprocessing_net_num_pool_op_kernel_sizes,
        fp16=False,
    )
    data_loader.initialize(not False)
    tr_gen = data_loader.tr_gen
    data_dict = next(tr_gen)
    data = data_dict["data"]
    target = data_dict["target"]
    print(data.shape)
    print(len(target))

    print(np.unique(data[0, 1]))
    b = 1
