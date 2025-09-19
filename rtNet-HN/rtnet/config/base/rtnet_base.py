#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from rtnet.utils import maybe_mkdir_p

from ..base.base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 1
        # number of channels in the very first conv layer
        self.in_channels = 1
        # resnet layers config
        self.backbone = "resnet18_3d"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_patch_size = [32, 118, 118]
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = None

        # --------------- transform config ----------------- #
        self.data_key = ("data",)
        self.label_key = "mask"
        # add more below per your need

        # --------------  training config --------------------- #
        # loss function for training
        self.loss_fn = "binary_cross_entropy_with_logit"
        # loss reduction method
        self.loss_reduction = "mean"
        # optimizer for training
        self.opt_setup = "SGD"
        # special config for optimizer
        self.opt_special_cfg = {"nesterov": True}
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "warmcos"  # apply EMA during training
        self.ema = False

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_input_patch_size = (32, 118, 118)

        # -------------- nnUnet style args, to be organized --------------#
        self.base = "/data/xxxx/datasets/LN_classify/rtNetData/"
        self.default_num_threads = 24
        # Data preprocessing, code will select median spacing if "target_spacing=None"
        # self.preproc_target_spacing = None
        self.preproc_target_spacing = [3, 0.46, 0.46]
        self.verify_preproc_integrity = False
        # Data loader tasks
        self.task_ids = ["004"]
        # --------------- Only support 2d, 3d_lowres, 3d_fullres --------------#
        self.preproc_resolution = "3d_fullres"
        self.current_fold = 0

        # Data manipulator

        self._do_not_touch()
        self._gen_data_paths()

    def _do_not_touch(self):
        # self.default_num_threads = min(self.default_num_threads, multiprocessing.cpu_count() - 2)
        self.preproc_resolutionample_sep_z_iso_thres = 3
        self.output_identifier = "rtNet"
        self.default_plans_identifier = "rtNetPlans"
        self.default_data_identifier = "rtNetData_plans"
        # self.default_data_loader = "rtNetDataLoader_SelectiveChannelDA"
        self.default_data_loader = "rtNetLNDataset"
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
            self.rtNet_raw_data = os.path.join(self.base_image, "rtNet_raw_data")
            self.rtNet_cropped_data = os.path.join(
                self.base_image, "rtNet_cropped_data"
            )
            maybe_mkdir_p(self.rtNet_raw_data)
            maybe_mkdir_p(self.rtNet_cropped_data)
        else:
            self.rtNet_cropped_data = self.rtNet_raw_data = None
        maybe_mkdir_p(
            self.preprocessing_output_dir
        ) if self.preprocessing_output_dir is not None else None
        if self.network_training_output_dir_base is not None:
            self.network_training_output_dir = os.path.join(
                self.network_training_output_dir_base, self.output_identifier
            )
            maybe_mkdir_p(self.network_training_output_dir)
        else:
            self.network_training_output_dir = None
        # Setup vector data planning & preprocessing paths
        maybe_mkdir_p(self.base_vector) if self.base_vector is not None else None
        # Setup scalar data planning & preprocessing paths
        maybe_mkdir_p(self.base_scalar) if self.base_scalar is not None else None

    def get_model(self):
        from rtnet.models import RtNet

        self.model = RtNet(
            backbone=self.backbone,
            num_classes=self.num_classes,
            in_channels=self.in_channels,
        )
        return self.model

    def get_data_loader(self, dataset, batch_size, is_distributed, no_aug=False):
        from rtnet.data import (
            DataLoader,
            InfiniteSampler,
            rtNetCectNcctDataset,
            worker_init_reset_seed,
        )
        from rtnet.data.data_manipulation import RandomCropWithSize

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        dataset = rtNetCectNcctDataset(
            dataset=dataset,
            preproc=RandomCropWithSize(
                self.input_patch_size,
                pad_mode="constant",
            ),
        )

        self.dataset = dataset

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["sampler"] = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        return iter(
            DataLoader(self.dataset, batch_size=batch_size, **dataloader_kwargs)
        )

    def get_loss_fn(self, loss_fn):
        from rtnet.loss.classification_loss import binary_cross_entropy_with_logit

        loss_fn = binary_cross_entropy_with_logit

        return loss_fn

    def get_optimizer(self, batch_size):
        # 推理版本不需要优化器
        raise NotImplementedError("Optimizer is not available in inference-only version")

    def get_lr_scheduler(self, lr, iters_per_epoch):
        # 推理版本不需要学习率调度器
        raise NotImplementedError("Learning rate scheduler is not available in inference-only version")

    def get_eval_loader(self, valdataset, batch_size, is_distributed):
        from rtnet.data import rtNetCectNcctDataset
        from rtnet.data.data_manipulation import RandomCropWithSize

        dataset = rtNetCectNcctDataset(
            dataset=valdataset,
            preproc=RandomCropWithSize(
                self.input_patch_size,
                pad_mode="constant",
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, dataset, batch_size, is_distributed, metric="auroc"):
        from rtnet.evaluators import BasicEvaluator

        val_loader = self.get_eval_loader(dataset, batch_size, is_distributed)
        evaluator = BasicEvaluator(
            dataloader=val_loader,
            img_size=self.test_input_patch_size,
            num_classes=self.num_classes,
            metric=metric,
        )
        return evaluator

    def get_trainer(self, args):
        from rtnet.run import Trainer

        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        return evaluator.evaluate(
            model, is_distributed, half, return_outputs=return_outputs
        )
