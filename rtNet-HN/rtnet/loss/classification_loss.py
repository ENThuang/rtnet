import torch
import torch.nn.functional as F


def binary_cross_entropy_with_logit_dummy_three_preds(input, target, *args, **kwargs):
    class_preds = input[:, 0]
    class_targets = target[:, 0]
    ene_preds = input[:, 1]
    ene_targets = target[:, 1]

    invalid_ene_mask = kwargs.pop("invalid_ene_mask")

    cls_loss = F.binary_cross_entropy_with_logits(
        class_preds, class_targets, *args, **kwargs
    )

    kwargs["reduction"] = "none"
    valid_mask = torch.logical_and(
        class_targets > 0, ~invalid_ene_mask
    )  # remove meta negative samples

    tmp_ene_loss = F.binary_cross_entropy_with_logits(
        ene_preds, ene_targets, *args, **kwargs
    )
    ene_loss = (tmp_ene_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return {
        "total_loss": cls_loss + ene_loss,
        "cls_loss": cls_loss,
        "ene_loss": ene_loss,
    }

def binary_cross_entropy_with_logit(input, target, *args, **kwargs):
    loss = F.binary_cross_entropy_with_logits(input, target, *args, **kwargs)
    return {"total_loss": loss}


def binary_cross_entropy_with_logit_dual_pred(input, target, *args, **kwargs):
    presv_loss = F.binary_cross_entropy_with_logits(
        input[:, 0].unsqueeze(-1), target, *args, **kwargs
    )
    invar_loss = F.binary_cross_entropy_with_logits(
        input[:, 1].unsqueeze(-1), target, *args, **kwargs
    )
    fused_loss = F.binary_cross_entropy_with_logits(
        input[:, 2].unsqueeze(-1), target, *args, **kwargs
    )
    loss = torch.mean(presv_loss + invar_loss + fused_loss)
    return {"total_loss": loss}


def binary_cross_entropy_with_logit_dual_pred_ene(input, target, *args, **kwargs):
    class_preds = input[:, ::2]
    class_targets = target[:, 0]
    ene_preds = input[:, 1::2]
    ene_targets = target[:, 1]

    invalid_ene_mask = kwargs.pop("invalid_ene_mask")

    presv_loss = F.binary_cross_entropy_with_logits(
        class_preds[:, 0], class_targets, *args, **kwargs
    )
    invar_loss = F.binary_cross_entropy_with_logits(
        class_preds[:, 1], class_targets, *args, **kwargs
    )
    fused_loss = F.binary_cross_entropy_with_logits(
        class_preds[:, 2], class_targets, *args, **kwargs
    )
    cls_loss = torch.mean(presv_loss + invar_loss + fused_loss)

    kwargs["reduction"] = "none"
    valid_mask = torch.logical_and(
        class_targets > 0, ~invalid_ene_mask
    )  # remove meta negative samples
    # valid_mask = ~invalid_ene_mask  # keep all meta negative samples

    tmp_presv_ene_loss = F.binary_cross_entropy_with_logits(
        ene_preds[:, 0], ene_targets, *args, **kwargs
    )
    # tmp_presv_ene_loss = sigmoid_focal_loss(ene_preds[:, 0], ene_targets, *args, **kwargs)
    presv_ene_loss = (tmp_presv_ene_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    tmp_invar_ene_loss = F.binary_cross_entropy_with_logits(
        ene_preds[:, 1], ene_targets, *args, **kwargs
    )
    # tmp_invar_ene_loss = sigmoid_focal_loss(ene_preds[:, 1], ene_targets, *args, **kwargs)
    invar_ene_loss = (tmp_invar_ene_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    tmp_fused_ene_loss = F.binary_cross_entropy_with_logits(
        ene_preds[:, 2], ene_targets, *args, **kwargs
    )
    # tmp_fused_ene_loss = sigmoid_focal_loss(ene_preds[:, 2], ene_targets, *args, **kwargs)
    fused_ene_loss = (tmp_fused_ene_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    ene_loss = torch.mean(presv_ene_loss + invar_ene_loss + fused_ene_loss)

    return {
        "total_loss": cls_loss + ene_loss,
        "cls_loss": cls_loss,
        "ene_loss": ene_loss,
    }


def binary_cross_entropy_with_logit_single_pred_ene(input, target, *args, **kwargs):
    class_preds = input[:, 0]
    class_targets = target[:, 0]
    ene_preds = input[:, 1]
    ene_targets = target[:, 1]

    invalid_ene_mask = kwargs.pop("invalid_ene_mask")

    cls_loss = F.binary_cross_entropy_with_logits(
        class_preds, class_targets, *args, **kwargs
    )

    kwargs["reduction"] = "none"
    valid_mask = torch.logical_and(
        class_targets > 0, ~invalid_ene_mask
    )  # remove meta negative samples
    # valid_mask = ~invalid_ene_mask  # keep all meta negative samples

    temp_ene_loss = F.binary_cross_entropy_with_logits(
        ene_preds, ene_targets, *args, **kwargs
    )
    ene_loss = (temp_ene_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return {
        "total_loss": cls_loss + ene_loss,
        "cls_loss": cls_loss,
        "ene_loss": ene_loss,
    }


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Adopted from PyTorch official. Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
