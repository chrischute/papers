import torch.nn as nn

from models.layers.bottleneck.base_bottleneck_block import BaseBottleneckBlock


class ResNetBottleneckBlock(BaseBottleneckBlock):

    expansion = 4

    def __init__(self, in_channels, num_channels, stride=1, use_pre_activation=False):
        """ResNeXt Basic Block. Two 3x3 convolutions with BatchNorm and activation.

        Args:
            in_channels: Number of channels in the input.
            num_channels: Number of channels in the basic block (output has num_channels * expansion).
            stride: Stride to use in the second convolutional layer.
            use_pre_activation: If True, use pre-activation ordering. Otherwise use the original formulation.
        """
        super(ResNetBottleneckBlock, self).__init__(in_channels, num_channels, stride, use_pre_activation)

    def _get_conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """Vanilla convolution for a vanilla ResNet.

        Note: `groups` gets ignored here.
        """
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        return conv
