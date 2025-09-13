import torch
import torch.nn as nn


class LogSumExpPool(nn.Module):
    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, D, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1, 1)
        """
        (N, C, D, H, W) = feat_map.shape

        # (N, C, 1, 1, 1) m
        m, _ = torch.max(feat_map.flatten(start_dim=2), dim=-1)
        m = m.view(N, C, 1, 1, 1)

        # (N, C, H, W) value0
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma

        # TODO: split dim=(-1, -2) for onnx.export
        return m + 1 / g * torch.log(
            area * torch.sum(torch.exp(g * value0), dim=(-1, -2, -3), keepdim=True)
        )
