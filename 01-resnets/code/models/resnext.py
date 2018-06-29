import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXt(nn.Module):
    """ResNeXt for CIFAR10.

    Based on the paper:
    "Aggregated Residual Transformations for Deep Neural Networks"
    by Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    (https://arxiv.org/abs/1611.05431).
    """
    def __init__(self):
        super(ResNeXt, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        pass
