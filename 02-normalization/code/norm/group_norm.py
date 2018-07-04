"""
Adapted from:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/normalization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters. Default: ``True``
        use_custom: If True, use custom implementation (e.g. to group temporal dim differently).
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, use_custom=False):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.use_custom = use_custom
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        if self.use_custom:
            return self._custom_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    @staticmethod
    def _custom_group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
        r"""Applies custom version of Group Normalization for last certain number of dimensions."""
        return torch.custom_group_norm(x, num_groups, weight, bias, eps,
                                       torch.backends.cudnn.enabled)
