#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import itertools
from os.path import exists, join
from os import makedirs
import cv2
import time
import numpy as np
from matplotlib import cm
from PIL import Image
from collections import ChainMap
from loguru import logger
from tqdm import tqdm
from rtnet.evaluators.cls_metrics import auroc

from functools import partial
from typing import List, Any, Union, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torchvision.transforms.functional as TF

from rtnet.utils import (
    gather,
    is_main_process,
    synchronize,
    time_synchronized,
)

from .basic_evaluator import BasicEvaluator


class Evaluator25D(BasicEvaluator):
    """
    Lymph node classification Evaluation class.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        num_classes: int,
        metric: str = "auroc",
        eval_25d_group: int = 1,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
        """
        super().__init__(dataloader, img_size, num_classes, metric)
        self.eval_25d_group = eval_25d_group

    def evaluate(self, model, distributed=False, half=False, return_outputs=False):
        """
        Lymph node Evaluation. Iterate inference on the test dataset.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            acc (float) : classification accuracy
            auc (float) : classification AUROC score
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        # image_wise_data = defaultdict(dict)
        image_wise_data = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, data_dict in enumerate(progress_bar(self.dataloader)):
            inps, labels = (
                data_dict["image_data"]["image"].cuda(),
                data_dict["image_data"]["label"].cuda(),
            )
            image_names = data_dict["image_data"]["image_name"]

            if "cls_index" in data_dict["image_data"]:
                cls_index = (
                    data_dict["image_data"]["cls_index"]
                    .to(torch.int8)
                    .unsqueeze(-1)
                    .cuda()
                )
            else:
                cls_index = (
                    torch.zeros(labels.shape[0]).to(torch.int8).unsqueeze(-1).cuda()
                )

            with torch.no_grad():
                inps = inps.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                # ins_preds = []
                # for bat_idx in range(inps.shape[0]):
                #     cur_data = inps[bat_idx]
                #     selected_slices = torch.nonzero(cur_data, as_tuple=True)[1].unique().tolist()
                #     assert len(selected_slices) > 0, "{} find no slice with fgr".format(image_names[bat_idx])
                #     slice_preds = []
                #     for slc_idx in selected_slices:
                #         cur_slice = cur_data[:, slc_idx, ...][None, ...]
                #         slice_preds.append(model(cur_slice))
                #     ins_preds.append(torch.cat(slice_preds).max().unsqueeze(-1))
                # preds = torch.cat(ins_preds)
                out = model(inps)
                if labels.shape[0] != out.shape[0]:
                    out = out.view(labels.shape[0], -1)
                    preds = out.max(dim=-1, keepdim=True)[0]
                else:
                    if out.shape[1] != 1:
                        preds = out.mean(dim=-1, keepdim=True)
                    else:
                        preds = out

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            for i, fname in enumerate(image_names):
                image_wise_data[fname] = {
                    "label": labels[i].cpu().item(),
                    "score": preds[i].cpu().item()
                    if preds.dim() == 1
                    else preds[i][-1].cpu().item(),
                }
            data_list.extend(
                [preds.unsqueeze(-1), labels.unsqueeze(-1), cls_index.cpu()]
            )

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])

        if distributed:
            data_list = gather(data_list, dst=0)
            image_wise_data = gather(image_wise_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            image_wise_data = dict(ChainMap(*image_wise_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, image_wise_data
        return eval_results


class Evaluator25DWithENE(Evaluator25D):
    """
    Lymph node classification Evaluation class.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        num_classes: int,
        metric: str = "auroc",
        eval_25d_group: int = 1,
        pred_ene: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
        """
        super().__init__(dataloader, img_size, num_classes, metric, eval_25d_group)
        self.pred_ene = pred_ene

    def evaluate_prediction(self, data_list, statistics):
        if not is_main_process():
            return {"overall": 0}, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].cpu().item()
        n_samples = statistics[1].cpu().item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward"],
                    [a_infer_time],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the cls comparing with the ground truth
        if len(data_list) > 0:
            preds = [item.cpu() for item in data_list[::3]]
            preds = torch.cat(preds, dim=0)
            labels = [item.cpu() for item in data_list[1::3]]
            labels = torch.cat(labels, dim=0)
            # cls_index = [item.cpu() for item in data_list[2::3]]
            # cls_index = torch.cat(cls_index, dim=0).squeeze(-1)

            eval_results = {}
            cls_pred = preds[:, 0]
            cls_label = labels[:, 0]
            cls_metric = eval(self.metric)(cls_pred, cls_label)
            metric_tag = self.metric + "_cls"
            eval_results[metric_tag] = cls_metric

            ene_pred = preds[:, 1]
            ene_label = labels[:, 1]
            if any(ene_label < 0):
                print("ENE label has uncertainty, partial ENE evaluation")
                ene_eval_mask = ene_label >= 0
                ene_pred = ene_pred[ene_eval_mask]
                ene_label = ene_label[ene_eval_mask]
                cls_label = cls_label[ene_eval_mask]
            ene_metric = eval(self.metric)(ene_pred, ene_label)
            metric_tag = self.metric + "_ene"
            eval_results[metric_tag] = ene_metric

            ene_pos_mask = cls_label > 0
            ene_pos_pred = ene_pred[ene_pos_mask]
            ene_pos_label = ene_label[ene_pos_mask]
            ene_metric = eval(self.metric)(ene_pos_pred, ene_pos_label)
            metric_tag = self.metric + "_ene_pos"
            eval_results[metric_tag] = ene_metric

            eval_results["overall"] = np.mean(list(eval_results.values()))

            info += "Mean evaluation %s is %.4f\n" % (
                self.metric,
                eval_results["overall"],
            )
            info += "Classification AUC is {:.4f}\n".format(
                eval_results[self.metric + "_cls"]
            )
            info += "ENE identification AUC is {:.4f}\n".format(
                eval_results[self.metric + "_ene"]
            )
            info += "ENE-pos identification AUC is {:.4f}".format(
                eval_results[self.metric + "_ene_pos"]
            )

            return eval_results, info
        else:
            return {"overall": 0}, info

    def evaluate(self, model, distributed=False, half=False, return_outputs=False):
        """
        Lymph node Evaluation. Iterate inference on the test dataset.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            acc (float) : classification accuracy
            auc (float) : classification AUROC score
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        # image_wise_data = defaultdict(dict)
        image_wise_data = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, data_dict in enumerate(progress_bar(self.dataloader)):
            inps, labels_cls = (
                data_dict["image_data"]["image"].cuda(),
                data_dict["image_data"]["label"].cuda(),
            )
            label_ene = data_dict["image_data"]["label_ene"].cuda()
            image_names = data_dict["image_data"]["image_name"]
            labels = torch.stack([labels_cls, label_ene], dim=1)

            if "cls_index" in data_dict["image_data"]:
                cls_index = (
                    data_dict["image_data"]["cls_index"]
                    .to(torch.int8)
                    .unsqueeze(-1)
                    .cuda()
                )
            else:
                cls_index = (
                    torch.zeros(labels.shape[0]).to(torch.int8).unsqueeze(-1).cuda()
                )

            with torch.no_grad():
                inps = inps.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                out = model(inps)
                cls_preds = out[:, 0::2].mean(dim=-1, keepdim=True)
                ene_preds = out[:, 1::2].mean(dim=-1, keepdim=True)
                # cls_preds = out[:, 4, None]
                # ene_preds = out[:, 5, None]
                preds = torch.cat([cls_preds, ene_preds], dim=-1)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            for i, fname in enumerate(image_names):
                image_wise_data[fname] = {
                    "cls_label": labels[i, 0].cpu().item(),
                    "cls_score": preds[i, 0].cpu().item(),
                    "ENE_label": labels[i, 1].cpu().item(),
                    "ENE_score": preds[i, 1].cpu().item(),
                }
                data_list.extend([preds, labels, cls_index.cpu()])

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])

        if distributed:
            data_list = gather(data_list, dst=0)
            image_wise_data = gather(image_wise_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            image_wise_data = dict(ChainMap(*image_wise_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, image_wise_data
        return eval_results


class Evaluator25DGradCAMWithENE(Evaluator25D):
    """
    Lymph node classification Evaluation class.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        num_classes: int,
        metric: str = "auroc",
        eval_25d_group: int = 1,
        pred_ene: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
        """
        super().__init__(dataloader, img_size, num_classes, metric, eval_25d_group)
        self.pred_ene = pred_ene
        self._hooks_enabled = True
        self.target_names = ["backbone_invariant.features.12.2"]
        self.cam_output_path = "/home/yirui.wang/Desktop/HN_LN_clinical_paper_CAM"

    def _store_grad(self, grad: Tensor, idx: int = 0) -> None:
        if self._hooks_enabled:
            self.hook_g[idx] = grad.data

    def _hook_a(
        self, module: nn.Module, input: Tensor, output: Tensor, idx: int = 0
    ) -> None:
        """Activation hook."""
        if self._hooks_enabled:
            self.hook_a[idx] = output.data

    def _hook_g(
        self, module: nn.Module, input: Tensor, output: Tensor, idx: int = 0
    ) -> None:
        """Gradient hook"""
        if self._hooks_enabled:
            self.hook_handles.append(
                output.register_hook(partial(self._store_grad, idx=idx))
            )

    def reset_hooks(self) -> None:
        """Clear stored activation and gradients."""
        self.hook_a: List[Optional[Tensor]] = [None] * len(self.target_names)
        self.hook_g: List[Optional[Tensor]] = [None] * len(self.target_names)

    def remove_hooks(self) -> None:
        """Clear model hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

        for handle in self.hook_back_handles:
            handle.remove()
        self.hook_back_handles.clear()

    def _backprop(
        self,
        scores: Tensor,
        class_idx: Union[int, List[int]],
        retain_graph: bool = False,
    ) -> None:
        """Backpropagate the loss for a specific output class"""

        # Backpropagate to get the gradients on the hooked layer
        if isinstance(class_idx, int):
            loss = scores[:, class_idx].sum()
        else:
            loss = scores.gather(
                1, torch.tensor(class_idx, device=scores.device).view(-1, 1)
            ).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)

    def _get_weights(
        self, class_idx: Union[int, List[int]], scores: Tensor, **kwargs: Any
    ) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""

        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)

        self.hook_g: List[Tensor]  # type: ignore[assignment]
        # Global average pool the gradients over spatial dimensions
        return [grad.flatten(2).mean(-1) for grad in self.hook_g]

    @staticmethod
    @torch.no_grad()
    def _normalize(
        cams: Tensor, spatial_dims: Optional[int] = None, eps: float = 1e-8
    ) -> Tensor:
        """CAM normalization."""
        spatial_dims = cams.ndim - 1 if spatial_dims is None else spatial_dims
        cams.sub_(
            cams.flatten(start_dim=-spatial_dims)
            .min(-1)
            .values[(...,) + (None,) * spatial_dims]
        )
        # Avoid division by zero
        cams.div_(
            cams.flatten(start_dim=-spatial_dims)
            .max(-1)
            .values[(...,) + (None,) * spatial_dims]
            + eps
        )

        return cams

    @staticmethod
    def overlay_mask(
        img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7
    ) -> Image.Image:
        """Overlay a colormapped mask on a background image
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> from torchcam.utils import overlay_mask
        >>> img = ...
        >>> cam = ...
        >>> overlay = overlay_mask(img, cam)
        Args:
            img: background image
            mask: mask to be overlayed in grayscale
            colormap: colormap to be applied on the mask
            alpha: transparency of the background image
        Returns:
            overlayed image
        Raises:
            TypeError: when the arguments have invalid types
            ValueError: when the alpha argument has an incorrect value
        """

        if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
            raise TypeError("img and mask arguments need to be PIL.Image")

        if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
            raise ValueError(
                "alpha argument is expected to be of type float between 0 and 1"
            )

        cmap = cm.get_cmap(colormap)
        # Resize mask and apply colormap
        overlay = mask.resize(img.size, resample=Image.BICUBIC)
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
        # Overlay the image with the mask
        overlayed_img = Image.fromarray(
            (alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8)
        )

        return overlayed_img

    def evaluate_prediction(self, data_list, statistics):
        if not is_main_process():
            return {"overall": 0}, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].cpu().item()
        n_samples = statistics[1].cpu().item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward"],
                    [a_infer_time],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the cls comparing with the ground truth
        if len(data_list) > 0:
            preds = [item.cpu() for item in data_list[::3]]
            preds = torch.cat(preds, dim=0)
            labels = [item.cpu() for item in data_list[1::3]]
            labels = torch.cat(labels, dim=0)
            # cls_index = [item.cpu() for item in data_list[2::3]]
            # cls_index = torch.cat(cls_index, dim=0).squeeze(-1)

            eval_results = {}
            cls_pred = preds[:, 0]
            cls_label = labels[:, 0]
            cls_metric = eval(self.metric)(cls_pred, cls_label)
            metric_tag = self.metric + "_cls"
            eval_results[metric_tag] = cls_metric

            ene_pred = preds[:, 1]
            ene_label = labels[:, 1]
            if any(ene_label < 0):
                print("ENE label has uncertainty, partial ENE evaluation")
                ene_eval_mask = ene_label >= 0
                ene_pred = ene_pred[ene_eval_mask]
                ene_label = ene_label[ene_eval_mask]
                cls_label = cls_label[ene_eval_mask]
            ene_metric = eval(self.metric)(ene_pred, ene_label)
            metric_tag = self.metric + "_ene"
            eval_results[metric_tag] = ene_metric

            ene_pos_mask = cls_label > 0
            ene_pos_pred = ene_pred[ene_pos_mask]
            ene_pos_label = ene_label[ene_pos_mask]
            ene_metric = eval(self.metric)(ene_pos_pred, ene_pos_label)
            metric_tag = self.metric + "_ene_pos"
            eval_results[metric_tag] = ene_metric

            eval_results["overall"] = np.mean(list(eval_results.values()))

            info += "Mean evaluation %s is %.4f\n" % (
                self.metric,
                eval_results["overall"],
            )
            info += "Classification AUC is {:.4f}\n".format(
                eval_results[self.metric + "_cls"]
            )
            info += "ENE identification AUC is {:.4f}\n".format(
                eval_results[self.metric + "_ene"]
            )
            info += "ENE-pos identification AUC is {:.4f}".format(
                eval_results[self.metric + "_ene_pos"]
            )

            return eval_results, info
        else:
            return {"overall": 0}, info

    def evaluate(self, model, distributed=False, half=False, return_outputs=False):
        """
        Lymph node Evaluation. Iterate inference on the test dataset.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            acc (float) : classification accuracy
            auc (float) : classification AUROC score
            summary (sr): summary info of evaluation.
        """
        # GradCAM
        self.model = model
        self.submodule_dict = dict(model.named_modules())
        self.reset_hooks()
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.hook_back_handles: List[torch.utils.hooks.RemovableHandle] = []

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        # Forward hook
        for idx, name in enumerate(self.target_names):
            self.hook_handles.append(
                self.submodule_dict[name].register_forward_hook(
                    partial(self._hook_a, idx=idx)
                )
            )
        # Backward hook
        for idx, name in enumerate(self.target_names):
            # Trick to avoid issues with inplace operations cf. https://github.com/pytorch/pytorch/issues/61519
            self.hook_back_handles.append(
                self.submodule_dict[name].register_forward_hook(
                    partial(self._hook_g, idx=idx)
                )
            )

        data_list = []
        # image_wise_data = defaultdict(dict)
        image_wise_data = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        with torch.set_grad_enabled(True):
            for cur_iter, data_dict in enumerate(progress_bar(self.dataloader)):
                inps, labels_cls = (
                    data_dict["image_data"]["image"].cuda(),
                    data_dict["image_data"]["label"].cuda(),
                )
                label_ene = data_dict["image_data"]["label_ene"].cuda()
                image_names = data_dict["image_data"]["image_name"]
                labels = torch.stack([labels_cls, label_ene], dim=1)

                if "cls_index" in data_dict["image_data"]:
                    cls_index = (
                        data_dict["image_data"]["cls_index"]
                        .to(torch.int8)
                        .unsqueeze(-1)
                        .cuda()
                    )
                else:
                    cls_index = (
                        torch.zeros(labels.shape[0]).to(torch.int8).unsqueeze(-1).cuda()
                    )

                inps = inps.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                model.zero_grad()
                out = model(inps)
                cls_preds = out[:, 0::2].mean(dim=-1, keepdim=True)
                ene_preds = out[:, 1::2].mean(dim=-1, keepdim=True)
                # cls_preds = out[:, 4, None]
                # ene_preds = out[:, 5, None]
                preds = torch.cat([cls_preds, ene_preds], dim=-1)

                # Grad-CAM
                weights = self._get_weights(1, preds)  # 1 for ENE
                for weight, activation in zip(weights, self.hook_a):
                    missing_dims = activation.ndim - weight.ndim  # type: ignore[union-attr]
                    weight = weight[(...,) + (None,) * missing_dims]

                    # Perform the weighted combination to get the CAM
                    cam = torch.nansum(weight * activation, dim=1)

                    # if do_relu:
                    cam = F.relu(cam, inplace=True)

                    # Normalize the CAM
                    # if normalized:
                    cam = self._normalize(cam, 3)

                    # resize CAM to match the input
                    cam = F.interpolate(
                        cam.unsqueeze(1), size=inps.shape[-2:], mode="bilinear"
                    )
                    cam = cam.view(-1, 3, cam.shape[-2], cam.shape[-1]).repeat(
                        1, 3, 1, 1
                    )

                    # TODO: save CAM
                    for sample_id in range(inps.shape[0]):
                        if "ene" not in image_names[sample_id].lower():
                            continue
                        for slice_id in range(inps.shape[3]):
                            cam_img = (
                                inps[sample_id, 0, 1, slice_id]
                                .cpu()
                                .numpy()
                                .astype(np.float)
                            )
                            cam_img = cam_img - cam_img.min()
                            cam_img = cam_img / (cam_img.max() + 1e-7)
                            cam_img = np.clip((255 * cam_img), 0, 255).astype(np.uint8)
                            cam_img = Image.fromarray(cam_img).convert("RGB")

                            cam_mask = cam[sample_id, slice_id].cpu()
                            cam_mask = TF.to_pil_image(cam_mask, mode="F")

                            composited = self.overlay_mask(cam_img, cam_mask)

                            # LN mask
                            # ln_mask = inps[sample_id, 0, 1, slice_id, ...].cpu().numpy()
                            # mask_img = np.clip(255 * ln_mask, 0, 255).astype(np.uint8)
                            # # Apply thresholding to the image
                            # ret, thresh = cv2.threshold(mask_img, 128, 255, cv2.THRESH_OTSU)
                            # # Find the contours in the image
                            # contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            # # Draw the obtained contour (or the set of coordinates forming a line) on the original image
                            # img_arr = np.asarray(cam_img)
                            # cv2.drawContours(img_arr, contours, -1, (0, 255, 0), 1)
                            # cam_img = Image.fromarray(img_arr)

                            sample_output_path = join(
                                self.cam_output_path, image_names[sample_id]
                            )
                            if not exists(sample_output_path):
                                makedirs(sample_output_path)
                            # dst = Image.new('RGB', (composited.width + cam_img.width, cam_img.height))
                            # dst.paste(composited, (0, 0))
                            # dst.paste(cam_img, (composited.width, 0))
                            # dst.save("{}/{}_{}.png".format(sample_output_path, image_names[sample_id], slice_id))
                            composited.save(
                                (
                                    "{}/{}_{}.png".format(
                                        sample_output_path,
                                        image_names[sample_id],
                                        slice_id,
                                    )
                                )
                            )

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                for i, fname in enumerate(image_names):
                    image_wise_data[fname] = {
                        "cls_label": labels[i, 0].cpu().item(),
                        "cls_score": preds[i, 0].cpu().item(),
                        "ENE_label": labels[i, 1].cpu().item(),
                        "ENE_score": preds[i, 1].cpu().item(),
                    }
                    data_list.extend([preds, labels, cls_index.cpu()])

        # clean data
        self.remove_hooks()
        self._hooks_enabled = False

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])

        if distributed:
            data_list = gather(data_list, dst=0)
            image_wise_data = gather(image_wise_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            image_wise_data = dict(ChainMap(*image_wise_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, image_wise_data
        return eval_results
