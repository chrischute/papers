import torch
import torch.nn as nn
import torch.nn.functional as F


class SEResNeXt(nn.Module):
    """SE-ResNeXt for CIFAR10.

    Based on the paper:
    "Squeeze-and-Excitation Networks"
    by Jie Hu, Li Shen, Gang Sun
    (https://arxiv.org/abs/1709.01507).
    """
    def __init__(self):
        super(SEResNeXt, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        pass
