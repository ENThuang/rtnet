#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep. 18 2022

@author: Dazhou Guo
"""

import os
import multiprocessing
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join


class Cfg(object):
    def __init__(self):
        # self.base = "/home/dazhou.guo/Projects/Data/rtNetData/"
        self.base = "/nas/yirui.wang/datasets/LN_classify/rtNetData/"
        self.default_num_threads = 24
        # Data preprocessing, code will select median spacing if "target_spacing=None"
        # self.preproc_target_spacing = None
        self.preproc_target_spacing = [3, 0.46, 0.46]
        self.verify_preproc_integrity = False
        # Data loader tasks
        self.task_ids = ["024"]
        # --------------- Only support 2d, 3d_lowres, 3d_fullres --------------#
        self.preproc_resolution = "3d_fullres"
        self.current_fold = 0
        # Data manipulator
        self.data_loader_planning_and_preprocessing_if_or_not_run_image_planning_and_preprocessing = (
            True
        )
        self.data_loader_planning_and_preprocessing_if_or_not_run_vector_planning_and_preprocessing = (
            False
        )
        self.data_loader_planning_and_preprocessing_if_or_not_run_scalar_planning_and_preprocessing = (
            False
        )
        # Data loader -- generate patch sizes for training
        self.data_loader_planning_and_preprocessing_batch_size = 2
        self.input_patch_size = [32, 96, 96]
        # --------------- For whole image reading, it needs "batch size=1" and "patch_size=None" --------------#
        # self.data_loader_planning_and_preprocessing_batch_size = 1
        # self.input_patch_size = None
        # --------------- For deep supervision, it will generate "GTs" based on different scales --------------#
        self.data_loader_planning_and_preprocessing_net_num_pool_op_kernel_sizes = [
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ]
        self._do_not_touch()
        self._gen_data_paths()

    def _do_not_touch(self):
        self.default_num_threads = min(
            self.default_num_threads, multiprocessing.cpu_count() - 2
        )
        self.preproc_resolutionample_sep_z_iso_thres = 3
        self.output_identifier = "rtNet"
        self.default_plans_identifier = "rtNetPlans"
        self.default_data_identifier = "rtNetData_plans"
        self.default_data_loader = "rtNetDataLoader_SelectiveChannelDA"
        # self.default_data_loader = "rtNetLNDataset"
        # Data loader -- image planning setup
        self.data_loader_planning_and_preprocessing_planner3d = "ExperimentPlanner3D"
        self.data_loader_planning_and_preprocessing_planner2d = "ExperimentPlanner2D"
        self.data_loader_planning_and_preprocessing_dont_run_preprocessing = False
        self.data_loader_planning_and_preprocessing_overwrite_plans = None
        self.data_loader_planning_and_preprocessing_overwrite_plans_identifier = None
        # Data loader -- thread setup
        self.data_loader_planning_and_preprocessing_num_of_threads_for_low_resolution_data_preprocessing = (
            self.default_num_threads
        )
        self.data_loader_planning_and_preprocessing_num_of_threads_for_full_resolution_data_preprocessing = (
            self.default_num_threads
        )
        # Data loader -- data aug -- dummy 2d data aug?
        self.do_lowres_planning = True
        self.do_dummy_2D_data_aug = False

    def _gen_data_paths(self):
        self.base_image = os.path.join(self.base, "images")
        self.base_vector = os.path.join(self.base, "vector")
        self.base_scalar = os.path.join(self.base, "scalar")
        # Setup image data planning & preprocessing paths
        self.preprocessing_output_dir = os.path.join(self.base_image, "preprocessed")
        self.network_training_output_dir_base = os.path.join(self.base_image, "results")
        if self.base_image is not None:
            self.rtNet_raw_data = join(self.base_image, "rtNet_raw_data")
            self.rtNet_cropped_data = join(self.base_image, "rtNet_cropped_data")
            maybe_mkdir_p(self.rtNet_raw_data)
            maybe_mkdir_p(self.rtNet_cropped_data)
        else:
            self.rtNet_cropped_data = self.rtNet_raw_data = None
        maybe_mkdir_p(
            self.preprocessing_output_dir
        ) if self.preprocessing_output_dir is not None else None
        if self.network_training_output_dir_base is not None:
            self.network_training_output_dir = join(
                self.network_training_output_dir_base, self.output_identifier
            )
            maybe_mkdir_p(self.network_training_output_dir)
        else:
            self.network_training_output_dir = None
        # Setup vector data planning & preprocessing paths
        maybe_mkdir_p(self.base_vector) if self.base_vector is not None else None
        # Setup scalar data planning & preprocessing paths
        maybe_mkdir_p(self.base_scalar) if self.base_scalar is not None else None


if __name__ == "__main__":
    cfg = Cfg()
