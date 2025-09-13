#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from abc import abstractmethod
from typing import List
from loguru import logger
import numpy as np

import torch

from rtnet.utils import is_main_process
from rtnet.evaluators.cls_metrics import auroc


class BasicEvaluator:
    """
    Lymph node classification Evaluation class.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        num_classes: int,
        cls_names: List = [],
        metric: str = "auroc",
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
        """

        self.dataloader = dataloader
        self.img_size = img_size
        self.num_classes = num_classes
        self.metric = metric
        self.cls_names = cls_names

    @abstractmethod
    def evaluate(self, model, distributed=False, half=False, return_outputs=False):
        """
        Lymph node Evaluation. Iterate inference on the test dataset.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            auc (float) : classification AUROC score
            summary (sr): summary info of evaluation.
        """
        pass

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
            cls_index = [item.cpu() for item in data_list[2::3]]
            cls_index = torch.cat(cls_index, dim=0).squeeze(-1)

            eval_results = {}
            eval_classes = cls_index.max().item() + 1
            for idx in range(eval_classes):
                cls_pred = preds[cls_index == idx, idx]
                cls_label = labels[cls_index == idx, idx]
                cls_metric = eval(self.metric)(cls_pred, cls_label)
                # use pre-defined class names if the num if class is equal
                # to the length of the given class names
                if eval_classes == len(self.cls_names) and idx < len(self.cls_names):
                    metric_tag = self.metric + "_{}".format(self.cls_names[idx])
                else:
                    metric_tag = self.metric + "_{}".format(idx)
                eval_results[metric_tag] = cls_metric

            if len(eval_results) > 1:
                overall_metric = np.mean(list(eval_results.values()))
                eval_results["overall"] = overall_metric
            else:
                deprecated_key = list(eval_results.keys())[0]
                eval_results["overall"] = eval_results.pop(deprecated_key)

            info += "Mean evaluation %s is %.4f" % (
                self.metric,
                eval_results["overall"],
            )

            return eval_results, info
        else:
            return {"overall": 0}, info
