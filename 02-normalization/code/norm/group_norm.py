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
        is_3d: If True, expect 3D inputs. Else expect 2D inputs.
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, use_custom=False, is_3d=False):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.is_3d = is_3d
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.use_custom = use_custom
        self.group_idxs = self._get_group_idxs(num_channels, num_groups)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()
        self.regroup()

    def regroup(self):
        """Regroup the channels into new groups."""
        # TODO: Keep track of useful stats and regroup properly. Right now we just match GroupNorm behavior.
        pass

    def forward(self, x):
        if self.use_custom:
            for group_idx in self.group_idxs:
                # Select the group of channels and normalize together
                group = torch.index_select(x, dim=1, index=group_idx)
                group_mean = torch.mean(group)
                group_var = torch.var(group)

                # Normalize
                for i in group_idx:
                    x[:, i] -= group_mean
                    x[:, i] /= torch.sqrt(group_var + self.eps)

            if self.affine:
                # Scale and shift
                if self.is_3d:
                    x = x * self.weight.view(-1, 1, 1, 1) + self.bias.view(-1, 1, 1, 1)
                else:
                    x = x * self.weight.view(-1, 1, 1) + self.bias.view(-1, 1, 1)

            return x
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    @staticmethod
    def _get_group_idxs(num_channels, num_groups):
        assert num_channels % num_groups == 0, 'Number of channels must be divisible by number of groups.'
        num_channels_per_group = num_channels // num_groups
        group_idxs = [nn.Parameter(torch.arange(i, i + num_channels_per_group, dtype=torch.int64), requires_grad=False)
                      for i in range(0, num_channels, num_channels_per_group)]
        group_idxs = nn.ParameterList(*group_idxs)

        return group_idxs
