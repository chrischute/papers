import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    """DenseNet for CIFAR-10.

    Based on the paper:
    "Densely Connected Convolutional Networks"
    by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    (https://arxiv.org/abs/1608.06993).
    """
    def __init__(self):
        super(DenseNet, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        pass
