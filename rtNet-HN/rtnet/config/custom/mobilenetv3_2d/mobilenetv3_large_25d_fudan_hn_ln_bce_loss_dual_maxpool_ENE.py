#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn

from rtnet.config.custom.mobilenetv3_2d.mobilenetv3_large_2d_fudan_hn_ln_bce_loss import (
    Exp as MyExp,
)


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.backbone = "mobilenet_v3_large"
        self.num_classes = 1
        self.pred_ene = True

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 8
        self.input_patch_size = [12, 60, 60]
        # self.input_patch_size = [12, 224, 224]
        # 2.5D data loading config
        self.slice_per_25d_group = 3
        self.num_25d_group = 1
        self.group_25d_overlap = 0

        # self.recist_lo = {'pos': None, 'neg': None}
        # self.recist_hi = {'pos': None, 'neg': None}
        self.recist_lo = None
        self.recist_hi = None

        # ------------ overrode default transform config ------------ #
        self.patch_center_dist_from_border = [p // 2 for p in self.input_patch_size]
        self.p_scale = 0.2
        self.p_rot = 0.2
        self.rotation_p_per_axis = 0.2

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_input_patch_size = (12, 60, 60)
        # self.test_input_patch_size = (12, 224, 224)
        self.test_num_25d_group = 1

        # -------------- nnUnet style args, to be organized --------------#
        # Data loader tasks
        self.base = "/nas/yirui.wang/datasets/LN_classify/rtNetData/"
        self.base_image = os.path.join(self.base, "images")
        self.preprocessing_output_dir = os.path.join(self.base_image, "preprocessed")
        self.task_ids = ["034"]
        self.task = "Task034_Fudan_HN_LN_ENE"
        self.processed_data_folder = os.path.join(
            self.preprocessing_output_dir, self.task
        )
        self.rtNet_raw_data = os.path.join(self.base_image, "rtNet_raw_data")
        self.rtNet_cropped_data = os.path.join(self.base_image, "rtNet_cropped_data")
        # --------------- Only support 2d, 3d_lowres, 3d_fullres --------------#
        self.preproc_resolution = "3d_fullres"
        self.current_fold = 0

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        print("Using {} for exp setup".format(os.path.basename(__file__)))

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

        # 2.5D settings
        self.data_aug_params["slice_per_25d_group"] = self.slice_per_25d_group
        self.data_aug_params["num_25d_group"] = self.num_25d_group
        self.data_aug_params["group_25d_overlap"] = self.group_25d_overlap

    def get_data_loader(self, dataset, batch_size, is_distributed):
        from rtnet.data import (
            rtNetLNClsDataset,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from rtnet.data.data_augment import get_rtnet_25d_dual_transforms

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        # data augmentations

        self.setup_DA_params()

        tr_augmentations = get_rtnet_25d_dual_transforms(
            params=self.data_aug_params,
            final_patch_size=self.input_patch_size,
            patch_center_dist_from_border=self.patch_center_dist_from_border,
            order_data=3,
            border_val_mask=-1,
            order_mask=1,
        )

        dataset = rtNetLNClsDataset(
            dataset=dataset,
            augmentations=tr_augmentations,
            is_2d=False,
            dual_input=True,
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
        from rtnet.evaluators import Evaluator25DWithENE

        if not is_test:
            val_loader = self.get_eval_loader(dataset, batch_size, is_distributed)
        else:
            val_loader = self.get_test_loader(dataset, batch_size, is_distributed)
        evaluator = Evaluator25DWithENE(
            dataloader=val_loader,
            img_size=self.test_input_patch_size,
            num_classes=self.num_classes,
            metric=metric,
            eval_25d_group=self.test_num_25d_group,
            pred_ene=self.pred_ene,
        )
        return evaluator

    def get_eval_loader(
        self,
        valdataset,
        batch_size,
        is_distributed,
    ):
        from rtnet.data import rtNetLNClsDataset
        from rtnet.data.data_augmentations.utility_transforms import NumpyToTensor
        from rtnet.data.data_augmentations.custom_transforms import (
            Make25dInput,
            DualInputInterpolateAndCrop,
        )

        # preprocessing
        dual_input_cropping = DualInputInterpolateAndCrop(
            self.test_input_patch_size, mask_index=1
        )
        to_tensor = NumpyToTensor(["data", "mask"], "float")
        make_25d_input = Make25dInput(
            slice_per_25d_group=self.slice_per_25d_group,
            num_25d_group=self.test_num_25d_group,
            group_25d_overlap=self.group_25d_overlap,
            is_test=True,
        )
        val_augmentations = [dual_input_cropping, make_25d_input, to_tensor]

        dataset = rtNetLNClsDataset(
            dataset=valdataset,
            augmentations=val_augmentations,
            is_2d=False,
            dual_input=True,
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

    def get_model(self):
        from rtnet.models import RtNet25DDualMILNetV2, RtNet25DDualMILNet
        from rtnet.models.backbones.mobilenetv3_2d import (
            MobileNet_V3_Small_Weights,
            MobileNet_V3_Large_Weights,
        )

        self.model = RtNet25DDualMILNet(
            backbone=self.backbone,
            pred_ene=self.pred_ene,
            num_classes=self.num_classes,
            num_25d_group=self.num_25d_group,
            slice_per_25d_group=self.slice_per_25d_group,
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        )
        #  weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
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
        from rtnet.data.data_augmentations.utility_transforms import NumpyToTensor
        from rtnet.data.data_augmentations.custom_transforms import (
            Make25dInput,
            DualInputInterpolateAndCrop,
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
        size_invar_cropping = DualInputInterpolateAndCrop(
            self.test_input_patch_size, mask_index=1
        )
        to_tensor = NumpyToTensor(["data", "mask"], "float")
        make_25d_input = Make25dInput(
            slice_per_25d_group=self.slice_per_25d_group,
            num_25d_group=self.test_num_25d_group,
            group_25d_overlap=self.group_25d_overlap,
            is_test=True,
        )
        test_augs = [
            resample_patient_spacing,
            intensity_norm,
            size_invar_cropping,
            make_25d_input,
            to_tensor,
        ]
        # get testing dataset
        test_dataset = rtNetLNClsNiftiDataset(
            dataset=test_data_list,
            dataset_properties=properties,
            augmentations=test_augs,
            is_2d=False,
            dual_input=True,
        )

        return test_dataset

    def get_test_data_files(self, base_folder):
        import json
        import csv

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
            lists.append(cur_pat)
        print("Total {} testing cases".format(len(lists)))

        return lists, {int(i): d["modality"][str(i)] for i in d["modality"].keys()}

    def get_loss_fn(self, loss_fn):
        from rtnet.loss.classification_loss import (
            binary_cross_entropy_with_logit_dual_pred_ene,
        )

        loss_fn = binary_cross_entropy_with_logit_dual_pred_ene

        return loss_fn
