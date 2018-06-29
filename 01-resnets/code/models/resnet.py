import torch.nn as nn

from models.base_resnet import BaseResNet
from models.layers import ResNetBasicBlock, ResNetBottleneckBlock


class ResNet(BaseResNet):
    """ResNet for CIFAR-10.

    Based on the paper:
    "Deep Residual Learning for Image Recognition"
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    (https://arxiv.org/abs/1512.03385).
    """

    depth2block = {18: ResNetBasicBlock,
                   34: ResNetBasicBlock,
                   50: ResNetBottleneckBlock,
                   101: ResNetBottleneckBlock,
                   152: ResNetBottleneckBlock}

    depth2config = {18: [2, 2, 2, 2],
                    34: [3, 4, 6, 3],
                    50: [3, 4, 6, 3],
                    101: [3, 4, 23, 3],
                    152: [3, 8, 36, 3]}

    def __init__(self, model_depth, init_channels, num_classes, use_pre_activation=True, **kwargs):
        """
        Args:
            model_depth: Depth of the ResNet model.
            init_channels: Number of channels in the initial residual block.
            num_classes: Number of classes on the output head.
            use_pre_activation: If true, use pre-activation order in the residual blocks.
        """
        super(ResNet, self).__init__(model_depth, init_channels, num_classes, use_pre_activation)
