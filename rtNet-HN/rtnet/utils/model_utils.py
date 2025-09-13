#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import contextlib
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn

__all__ = [
    "get_model_info",
    "replace_module",
    "freeze_module",
    "adjust_status",
]


def get_model_info(model: nn.Module, tsize: Sequence[int], nchannels: int = 1) -> str:
    # Use thop tool to count the FLOPs of PyTorch model. The calculation method can
    # be found in "Pruning Convolutional Neural Networks for Resource Efficient Inference"
    # https://arxiv.org/pdf/1611.06440.pdf Appendix A.1.
    # Note: FLOPs = 2 x MAC (Multiply–accumulate operation)

    from thop import profile

    # TODO: modify the input size to accomendate CT 3D patches
    stride = 64
    img = torch.zeros(
        (1, nchannels, stride, stride, stride), device=next(model.parameters()).device
    )
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] * tsize[2] / stride / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def replace_module(
    module, replaced_module_type, new_module_type, replace_func=None
) -> nn.Module:
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


def freeze_module(module: nn.Module, name=None) -> nn.Module:
    """freeze module inplace

    Args:
        module (nn.Module): module to freeze.
        name (str, optional): name to freeze. If not given, freeze the whole module.
            Note that fuzzy match is not supported. Defaults to None.

    Examples:
        freeze the backbone of model
        >>> freeze_moudle(model.backbone)

        or freeze the backbone of model by name
        >>> freeze_moudle(model, name="backbone")
    """
    for param_name, parameter in module.named_parameters():
        if name is None or name in param_name:
            parameter.requires_grad = False

    # ensure module like BN and dropout are freezed
    for module_name, sub_module in module.named_modules():
        # actually there are no needs to call eval for every single sub_module
        if name is None or name in module_name:
            sub_module.eval()

    return module


@contextlib.contextmanager
def adjust_status(module: nn.Module, training: bool = False) -> nn.Module:
    """Adjust module to training/eval mode temporarily.

    Args:
        module (nn.Module): module to adjust status.
        training (bool): training mode to set. True for train mode, False fro eval mode.

    Examples:
        >>> with adjust_status(model, training=False):
        ...     model(data)
    """
    status = {}

    def backup_status(module):
        for m in module.modules():
            # save prev status to dict
            status[m] = m.training
            m.training = training

    def recover_status(module):
        for m in module.modules():
            # recover prev status from dict
            m.training = status.pop(m)

    backup_status(module)
    yield module
    recover_status(module)
