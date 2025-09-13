#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import re
import warnings

import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from rtnet.config import get_exp
from rtnet.run import launch
from rtnet.utils import (
    configure_module,
    configure_nccl,
    get_local_rank,
    get_model_info,
    setup_logger,
)


def make_parser():
    parser = argparse.ArgumentParser("RTNet Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    # parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="test_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    # if args.tsize is not None:
    #     exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    # load preprocess plan and preprocess nifti data
    dataset_test = exp.get_test_data()
    evaluator = exp.get_evaluator(
        dataset=dataset_test,
        batch_size=args.batch_size,
        is_distributed=is_distributed,
        metric="auroc",
        is_test=True,
    )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint from {}".format(ckpt_file))
    loc = "cuda:{}".format(rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    state_dict = {}
    for k, v in ckpt["model"].items():
        # remove the prefix 'module.' introduced by DDP
        new_key = re.sub("^module.", "", k)
        state_dict[new_key] = v
    model.load_state_dict(state_dict)
    logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # start evaluate
    (metric, summary), predictions = evaluator.evaluate(
        model, is_distributed, args.fp16, return_outputs=True
    )

    # save prediction scores and labels
    with open(os.path.join(file_name, "predictions.txt"), "w") as f:
        for k, v in predictions.items():
            if "pred_ene" not in exp.__dict__ or not exp.pred_ene:
                f.write("{},{},{}".format(k, v["label"], v["score"]))
                f.write("\n")
            else:
                f.write(
                    "{},{},{},{},{}".format(
                        k,
                        v["cls_label"],
                        v["cls_score"],
                        v["ENE_label"],
                        v["ENE_score"],
                    )
                )
                f.write("\n")

    logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
