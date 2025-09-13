#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep. 18 2022

@author: Dazhou Guo
"""
from rtnet.data.data_loader.data_image_utils.utils_data_loading import *
from rtnet.data.data_loader.data_image_utils.utils import convert_id_to_task_name
from rtnet.data.data_loader.cfg_base import Cfg

from typing import List

cfg = Cfg()


class DataImageLoader(object):
    def __init__(
        self,
        res=cfg.preproc_resolution,
        task=cfg.task_ids,
        fold=cfg.current_fold,
        data_loader=cfg.default_data_loader,
        default_plans_identifier=cfg.default_plans_identifier,
        input_patch_size=cfg.input_patch_size,
        net_num_pool_op_kernel_sizes=cfg.data_loader_planning_and_preprocessing_net_num_pool_op_kernel_sizes,
        fp16=False,
    ):
        self.input_patch_size = input_patch_size
        self.fold = fold
        self.net_num_pool_op_kernel_sizes = net_num_pool_op_kernel_sizes
        self.fp16 = fp16
        if isinstance(task, List):
            if isinstance(int(task[0]), int):
                self.task = convert_id_to_task_name(int(task[0]))
            else:
                self.task = task[0]
        elif isinstance(task, str):
            self.task = convert_id_to_task_name(int(task))
        else:
            raise ValueError("task must be a string or a list of strings or integers")

        (
            self.plans_file,
            self.output_folder_name,
            self.dataset_directory,
            self.stage,
            self.data_loader_class,
        ) = get_default_configuration(
            res, self.task, data_loader, default_plans_identifier
        )

    def initialize(self, is_training=True, force_load_plans=False):
        data_loader = self.data_loader_class(
            self.plans_file,
            self.input_patch_size,
            self.fold,
            output_folder=self.output_folder_name,
            dataset_directory=self.dataset_directory,
            stage=self.stage,
            unpack_data=True,
            deterministic=True,
            net_num_pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
            fp16=self.fp16,
        )
        data_loader.initialize(is_training, force_load_plans)
        return data_loader.tr_gen, data_loader.val_gen


if __name__ == "__main__":
    dl = DataImageLoader()
    tr_gen, val_gen = dl.initialize()

    data_dict = next(tr_gen)
    data = data_dict["data"]
    target = data_dict["target"]
    print(data.shape)
    print(len(target))
