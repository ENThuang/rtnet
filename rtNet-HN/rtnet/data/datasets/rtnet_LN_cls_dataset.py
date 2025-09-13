#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep. 18 2022

@author: Dazhou Guo
"""
import os
from typing import List, Sequence
from collections import OrderedDict
import numpy as np
import nibabel as nib
import SimpleITK as sitk

import torch
import torchvision
from torch.utils.data.dataset import Dataset as torchDataset
from torchvision.utils import make_grid

from rtnet.data.utils import load_pickle
from rtnet.data.data_augmentations import Compose


class rtNetLNClsDataset(torchDataset):
    def __init__(
        self,
        dataset,
        augmentations=None,
        memmap_mode="r",
        is_2d=False,
        drop_top_and_bottom=False,
        dual_input=False,
    ):
        self.dataset = dataset
        self.data_list = list(dataset.keys())
        if augmentations is not None:
            if not isinstance(augmentations, Sequence):
                augmentations = [augmentations]
            augmentations = Compose(augmentations)
        self.augmentations = augmentations
        self.memmap_mode = memmap_mode
        self.is_2d = is_2d
        self.dual_input = dual_input
        self.drop_top_and_bottom = drop_top_and_bottom

    def __getitem__(self, idx):
        data_file_name = self.data_list[idx]
        if "properties" in self.dataset[data_file_name].keys():
            properties = self.dataset[data_file_name]["properties"]
        else:
            properties = load_pickle(self.dataset[data_file_name]["properties_file"])

        # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
        # which is much faster to access
        if os.path.isfile(self.dataset[data_file_name]["data_file"][:-4] + ".npy"):
            image_data = np.load(
                self.dataset[data_file_name]["data_file"][:-4] + ".npy",
                self.memmap_mode,
            )
        else:
            image_data = np.load(self.dataset[data_file_name]["data_file"])["data"]

        image_name = (
            self.dataset[data_file_name]["data_file"][:-4].lower().split("/")[-1]
        )

        if self.is_2d:
            # random select slice
            image_data = self.random_slice(
                image_data, self.drop_top_and_bottom
            )  # (mod, d, h, w)->(mod, h, w)

        label = 1 if "pos" in image_name else 0

        label_ene = 1 if "_ene" in image_name.lower() else 0
        # the ENE label of positive samples in the first batch is unknown
        valid_ene = (
            "ZS" in image_name.upper()
            or "ext_" in image_name.lower()
            or "int_" in image_name.lower()
        )
        if not valid_ene and label == 1:
            label_ene = -1

        if self.augmentations:
            data_dict = {
                "data": np.expand_dims(image_data[0], axis=(0, 1)),
                # 'mask': np.expand_dims(image_data[1], axis=(0, 1))
                "mask": np.expand_dims(image_data[1:], axis=0),
            }
            """
            the input should have a shape of (mod, chn, d, h, w) for 3d and a shape of (mod, chn, h, w) for 2d slice.
            if multiple masks/samples are provided at the same time, concatenate them into the channel dimension per
            data augmentation's operation requirements.
            """
            data_dict = self.augmentations(**data_dict)
            if not self.dual_input:
                image_data = data_dict["data"][0]
                dilated_mask = data_dict["mask"][1]
                raw_mask = data_dict["mask"][0]
            else:
                image_data = data_dict["data"].permute(1, 0, 2, 3, 4)
                dilated_mask = data_dict["mask"][1::2].permute(1, 0, 2, 3, 4)
                raw_mask = data_dict["mask"][0::2].permute(1, 0, 2, 3, 4)
        else:
            raise NotImplementedError

        # correct label is the mask is all zero
        if dilated_mask.sum() == 0:
            if label == 1:
                label = 0
                label_ene = 0
                print(
                    "{} label was changed to NEGATIVE due to empty mask.".format(
                        image_name
                    )
                )
            else:
                print("{} has no LN mask. Skipped augmentations".format(image_name))

        if self.is_2d:
            assert len(image_data.shape) == 3
        else:
            if not self.dual_input:
                assert len(image_data.shape) == 4
            else:
                assert len(image_data.shape) == 5

        # mask out non-ROI regions
        image_data *= dilated_mask

        return {
            "image_data": {
                "image": image_data,
                "label": label,
                "label_ene": label_ene,
                "image_name": image_name,
            },
            # "vector": raw_label_mask,  # dummy value
            "vector": idx,  # dummy value
            "scalar": idx,  # dummy value
        }

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def random_slice(data, drop_top_and_bottom=False):
        image = data[0].copy()
        dilated_mask = data[2].copy()
        raw_mask = data[1].copy()
        # get nonzero slice indices
        nonzero_slice_idx = np.unique(np.nonzero(raw_mask)[0]).tolist()
        if drop_top_and_bottom and len(nonzero_slice_idx) > 3:
            nonzero_slice_idx = nonzero_slice_idx[1:-1]
        rnd_slice = np.random.choice(
            nonzero_slice_idx
        )  # use raw mask to select a slice
        sliced_data = np.vstack(
            (
                image[rnd_slice][None, ...],
                raw_mask[rnd_slice][None, ...],
                dilated_mask[rnd_slice][None, ...],
            )
        )

        return sliced_data


class rtNetLNClsNiftiDataset(torchDataset):
    def __init__(
        self,
        dataset,
        dataset_properties,
        augmentations=None,
        memmap_mode="r",
        is_2d=False,
        attach_LN_mask=False,
        keep_LNs_only=False,
        dual_input=False,
    ):
        self.dataset = dataset
        self.dataset_properties = dataset_properties
        if augmentations is not None:
            if not isinstance(augmentations, Sequence):
                augmentations = [augmentations]
            augmentations = Compose(augmentations)
        self.augmentations = augmentations
        self.memmap_mode = memmap_mode
        self.is_2d = is_2d
        self.attach_LN_mask = attach_LN_mask
        self.keep_LNs_only = keep_LNs_only
        self.dual_input = dual_input

    def __getitem__(self, idx):
        data_files = self.dataset[idx]
        data_itk, properties = self.get_data_and_properties(data_files)
        image_data = np.vstack(
            [sitk.GetArrayFromImage(d)[None] for d in data_itk]
        ).astype(np.float32)
        image_name = data_files[0][:-12].lower().split("/")[-1]
        label = 1 if "pos" in image_name else 0
        label_ene = 1 if "_ene" in image_name.lower() else 0
        # the ENE label of positive samples in the first batch is unknown
        valid_ene = (
            "ZS" in image_name.upper()
            or "ext_" in image_name.lower()
            or "int_" in image_name.lower()
        )
        if not valid_ene and label == 1:
            label_ene = -1

        if self.augmentations:
            data_dict = {
                "data": np.expand_dims(image_data[0], axis=(0, 1)),
                "mask": np.expand_dims(image_data[1:], axis=0),
                "properties": properties,
                "intensityproperties": self.dataset_properties["intensityproperties"],
            }
            """
            the input should have a shape of (mod, chn, d, h, w) for 3d and a shape of (mod, chn, h, w) for 2d slice.
            if multiple masks/samples are provided at the same time, concatenate them into the channel dimension per
            data augmentation's operation requirements.
            """
            data_dict = self.augmentations(**data_dict)
            if not self.dual_input:
                image_data = data_dict["data"][0]
                dilated_mask = data_dict["mask"][1]
                raw_mask = data_dict["mask"][0]
            else:
                image_data = data_dict["data"].permute(1, 0, 2, 3, 4)
                dilated_mask = data_dict["mask"][1::2].permute(1, 0, 2, 3, 4)
                raw_mask = data_dict["mask"][0::2].permute(1, 0, 2, 3, 4)
        else:
            raise NotImplementedError

        if self.is_2d:
            assert len(image_data.shape) == 3
        else:
            if not self.dual_input:
                assert len(image_data.shape) == 4
            else:
                assert len(image_data.shape) == 5

        # mask out non-ROI regions
        image_data *= dilated_mask

        # correct label is the mask is all zero
        if dilated_mask.sum() == 0:
            if label == 1:
                label = 0
                label_ene = 0
                print(
                    "{} label was changed to NEGATIVE due to empty mask.".format(
                        image_name
                    )
                )
            else:
                print("{} has no LN mask. Skipped augmentations".format(image_name))

        return {
            "image_data": {
                "image": image_data,
                "label": label,
                "label_ene": label_ene,
                "image_name": image_name,
            },
            # "vector": raw_label_mask,  # dummy value
            "vector": idx,  # dummy value
            "scalar": idx,  # dummy value
        }

    def get_data_and_properties(self, data_files):
        data_itk = [sitk.ReadImage(f) for f in data_files]
        properties = OrderedDict()

        properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[
            [2, 1, 0]
        ]
        properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
        properties["list_of_data_files"] = data_files

        properties["itk_origin"] = data_itk[0].GetOrigin()
        properties["itk_spacing"] = data_itk[0].GetSpacing()
        properties["itk_direction"] = data_itk[0].GetDirection()

        return data_itk, properties

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    # TODO: add test sample here
    pass
