#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn

from rtnet.config.base.rtnet_base import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 1
        self.in_channels = 1
        # resnet layers config
        self.backbone = "mobilenet_v3_large"
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_patch_size = [60, 60]
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = None
        # lower bound of RECIST for loading lymph nodes
        self.recist_lo = {"pos": None, "neg": None}
        # self.recist_lo = {'pos': None, 'neg': -1}
        # upper bound of RECIST for loading lymph nodes
        # self.recist_hi = {'pos': 10, 'neg': None}
        self.recist_hi = {"pos": None, "neg": None}
        self.drop_top_and_bottom = False

        # ------------ overrode default transform config ------------ #
        self.patch_center_dist_from_border = [p // 2 for p in self.input_patch_size]
        self.p_scale = 0.2
        self.p_rot = 0.2
        self.rotation_p_per_axis = 0.2

        # --------------  training config --------------------- #
        # loss function for training
        self.loss_fn = "binary_cross_entropy_with_logit"
        # loss reduction method
        self.loss_reduction = "mean"
        # optimizer for training
        self.opt_setup = "Adam"
        # special config for optimizer
        self.opt_special_cfg = {}
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
        self.scheduler = "constant"

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 5
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 5
        # save history checkpoint or not.
        # If set to False, rtnet will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_input_patch_size = (32, 60, 60)

        # -------------- nnUnet style args, to be organized --------------#
        # Data loader tasks
        self.task_ids = ["007"]
        self.task = "Task007_Fudan_HN_LN_270_cases"
        self.processed_data_folder = os.path.join(
            self.preprocessing_output_dir, self.task
        )
        # --------------- Only support 2d, 3d_lowres, 3d_fullres --------------#
        self.preproc_resolution = "2d"
        self.current_fold = 0

    def setup_DA_params(self):
        from rtnet.data.data_augment import (
            default_3D_augmentation_params,
            default_2D_augmentation_params,
        )

        threeD = True if self.preproc_resolution != "2d" else False

        if threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params["rotation_x"] = (
                -45.0 / 360 * 2.0 * np.pi,
                45.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_y"] = (
                -45.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_z"] = (
                -15.0 / 360 * 2.0 * np.pi,
                15.0 / 360 * 2.0 * np.pi,
            )
        else:
            if max(self.input_patch_size) / min(self.input_patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (
                    -15.0 / 360 * 2.0 * np.pi,
                    15.0 / 360 * 2.0 * np.pi,
                )
            self.data_aug_params = default_2D_augmentation_params

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["selected_seg_channels"] = [0]
        self.data_aug_params["patch_size_for_spatialtransform"] = self.input_patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

        # override default augmentation params
        if "p_scale" in self.__dict__:
            self.data_aug_params["p_scale"] = self.p_scale
        if "p_rot" in self.__dict__:
            self.data_aug_params["p_rot"] = self.p_rot
        if "rotation_p_per_axis" in self.__dict__:
            self.data_aug_params["rotation_p_per_axis"] = self.rotation_p_per_axis

    def get_data_loader(self, dataset, batch_size, is_distributed):
        from rtnet.data import (
            rtNetLNClsDataset,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from rtnet.data.data_augment import get_rtnet_transforms

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        # data augmentations

        self.setup_DA_params()

        tr_augmentations = get_rtnet_transforms(
            params=self.data_aug_params,
            final_patch_size=self.input_patch_size,
            patch_center_dist_from_border=self.patch_center_dist_from_border,
            order_data=3,
            border_val_mask=-1,
            order_mask=1,
        )

        dataset = rtNetLNClsDataset(
            dataset=dataset, augmentations=tr_augmentations, is_2d=True
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

    def get_evaluator(
        self, dataset, batch_size, is_distributed, metric="auroc", is_test=False
    ):
        from rtnet.evaluators import PerSlice2DEvaluator

        if not is_test:
            val_loader = self.get_eval_loader(dataset, batch_size, is_distributed)
        else:
            val_loader = self.get_test_loader(dataset, batch_size, is_distributed)
        evaluator = PerSlice2DEvaluator(
            dataloader=val_loader,
            img_size=self.test_input_patch_size,
            num_classes=self.num_classes,
            metric=metric,
        )
        return evaluator

    def get_eval_loader(
        self,
        valdataset,
        batch_size,
        is_distributed,
    ):
        from rtnet.data import rtNetLNClsDataset
        from rtnet.data.data_manipulation import CenterCropAndMaskOut
        from rtnet.data.data_augmentations.utility_transforms import NumpyToTensor
        from rtnet.data.data_augmentations.custom_transforms import (
            MakePseudoRGBInputFrom3D,
        )

        # preprocessing
        center_crop_with_sz = CenterCropAndMaskOut(
            self.test_input_patch_size,
            pad_mode="constant",
            is_2d=False,  # use pseudo 3d to eval all the slices
        )
        to_tensor = NumpyToTensor(["data", "mask"], "float")
        make_pseudi_rgb_from_3d = MakePseudoRGBInputFrom3D()
        val_augmentations = [center_crop_with_sz, make_pseudi_rgb_from_3d, to_tensor]

        dataset = rtNetLNClsDataset(
            dataset=valdataset,
            augmentations=val_augmentations,
            # is_2d=self.preproc_resolution == '2d'
            is_2d=False,  # use pseudo 3d to eval all the slices
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

    def get_loss_fn(self, loss_fn):
        from rtnet.loss.classification_loss import binary_cross_entropy_with_logit

        loss_fn = binary_cross_entropy_with_logit

        return loss_fn

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            if self.opt_setup == "Adam":
                optimizer = eval(f"torch.optim.{self.opt_setup}")(
                    pg0, lr=lr, **self.opt_special_cfg
                )
            elif self.opt_setup == "SGD":
                optimizer = eval(f"torch.optim.{self.opt_setup}")(
                    pg0, lr=lr, momentum=self.momentum, **self.opt_special_cfg
                )
            else:
                raise NotImplementedError

            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_model(self):
        from rtnet.models import RtNet
        from rtnet.models.backbones.mobilenetv3_2d import MobileNet_V3_Large_Weights

        self.model = RtNet(
            backbone=self.backbone,
            num_classes=self.num_classes,
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        )
        return self.model

    def get_test_loader(self, dataset, batch_size, is_distributed):
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

    def get_test_data(self):
        from rtnet.data import rtNetLNClsNiftiDataset
        from rtnet.data.data_manipulation import CenterCropAndMaskOut
        from rtnet.data.data_augmentations.utility_transforms import NumpyToTensor
        from rtnet.data.data_augmentations.custom_transforms import (
            MakePseudoRGBInputFrom3D,
        )
        from rtnet.data.data_augmentations.resample_transforms import (
            ResamlePixelSpacing,
        )
        from rtnet.data.data_augmentations.intensity_transforms import (
            IntensityNormalization,
        )
        from rtnet.data.utils import (
            load_pickle,
            determine_normalization_scheme,
            convert_id_to_task_name,
        )

        task = convert_id_to_task_name(
            int(self.task_ids[0]),
            self.preprocessing_output_dir,
            self.rtNet_raw_data,
            self.rtNet_cropped_data,
            self.network_training_output_dir,
        )
        # list testing samples
        base_folder = os.path.join(self.rtNet_raw_data, task)
        test_data_list, _ = self.get_test_data_files(base_folder)
        # get pixel spacing resamlple transform
        properties = load_pickle(
            os.path.join(self.rtNet_cropped_data, task, "dataset_properties.pkl")
        )
        resample_patient_spacing = ResamlePixelSpacing(properties)
        # get normalization transform
        normalization_schemes = determine_normalization_scheme(properties)
        intensity_norm = IntensityNormalization(normalization_schemes)
        # center crop transform
        center_crop_with_sz = CenterCropAndMaskOut(
            self.test_input_patch_size,
            pad_mode="constant",
            is_2d=False,  # use pseudo 3d to eval all the slices
        )
        to_tensor = NumpyToTensor(["data", "mask"], "float")
        make_pseudi_rgb_from_3d = MakePseudoRGBInputFrom3D()
        test_augs = [
            resample_patient_spacing,
            intensity_norm,
            center_crop_with_sz,
            make_pseudi_rgb_from_3d,
            to_tensor,
        ]
        # get testing dataset
        test_dataset = rtNetLNClsNiftiDataset(
            dataset=test_data_list,
            dataset_properties=properties,
            augmentations=test_augs,
            # is_2d=self.preproc_resolution == '2d'
            is_2d=False,  # use pseudo 3d to eval all the slices
        )

        return test_dataset

    def get_test_data_files(self, base_folder):
        import json
        import csv

        # load instance recist info
        ins_info = {}
        pos_cnt, neg_cnt = 0, 0
        with open(
            os.path.join(self.processed_data_folder, "cropping_list.csv"), "r"
        ) as data:
            for line in csv.DictReader(data):
                s_pth = line["path"].replace("/", "_")
                s_basename = line["basename"]
                filename = s_pth + "_" + s_basename if s_pth else s_basename
                filename += "_pos" if int(line["label"]) == 1 else "_neg"
                ins_info[filename] = {"recist": line["recist"], "label": line["label"]}

        json_file = os.path.join(base_folder, "dataset.json")
        with open(json_file) as jsn:
            d = json.load(jsn)
            testing_files = d["test"]
        num_modalities = len(d["modality"].keys())

        # adapted from data loader utils 'create_lists_from_splitted_dataset'
        lists = []
        for ts in testing_files:
            cur_pat = []
            for mod in range(num_modalities):
                cur_pat.append(
                    os.path.join(
                        base_folder, ts.replace("./", "")[:-7] + "_%04.0d.nii.gz" % mod
                    )
                )

            # filter out ins
            c = ts.split("/")[-1][:-7]
            if int(ins_info[c]["label"]) == 0:
                if (
                    self.recist_lo["neg"] is not None
                    and float(ins_info[c]["recist"]) < self.recist_lo["neg"]
                ):
                    continue
            else:
                if (
                    self.recist_hi["pos"] is not None
                    and float(ins_info[c]["recist"]) > self.recist_hi["pos"]
                ):
                    continue

            if int(ins_info[c]["label"]) == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1

            lists.append(cur_pat)
        print("Total {} testing cases".format(len(lists)))
        print("Positive {} cases, negative {} cases".format(pos_cnt, neg_cnt))

        return lists, {int(i): d["modality"][str(i)] for i in d["modality"].keys()}
