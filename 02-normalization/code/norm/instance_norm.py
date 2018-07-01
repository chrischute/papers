import torch.nn as nn


class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNorm2d, self).__init__()

    def forward(self, x):
        raise NotImplementedError
