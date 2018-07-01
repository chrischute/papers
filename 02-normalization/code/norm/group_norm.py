import torch.nn as nn


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_features):
        super(GroupNorm, self).__init__()

    def forward(self, x):
        raise NotImplementedError
