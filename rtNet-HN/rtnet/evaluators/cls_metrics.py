import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score


def accuracy(pred, target, cutoff=0.5):
    if pred.device != torch.device("cpu"):
        pred = pred.to("cpu")
    if target.device != torch.device("cpu"):
        target = target.to("cpu")

    if torch.is_tensor(pred):
        pred = pred.numpy()
    if torch.is_tensor(target):
        target = target.numpy()

    binary_pred = 1.0 * (pred >= cutoff).flatten()
    hit = sum((binary_pred == target) * 1.0)
    total = len(binary_pred)
    acc = hit / total

    return acc


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def auroc(pred, target):
    if pred.device != torch.device("cpu"):
        pred = pred.to("cpu")
    if target.device != torch.device("cpu"):
        target = target.to("cpu")

    if torch.is_tensor(pred):
        pred = pred.numpy()
    if torch.is_tensor(target):
        target = target.numpy()

    fpr, tpr, thres = roc_curve(target.flatten(), pred.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc
