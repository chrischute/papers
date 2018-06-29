import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """ResNet for CIFAR-10.

    Based on the paper:
    "Deep Residual Learning for Image Recognition"
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    (https://arxiv.org/abs/1512.03385).
    """
    def __init__(self):
        super(ResNet, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        pass
