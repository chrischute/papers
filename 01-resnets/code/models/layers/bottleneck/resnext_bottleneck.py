import torch.nn as nn

from .base_bottleneck import BaseBottleneck


class ResNeXtBottleneck(BaseBottleneck):

    expansion = 2

    def __init__(self, in_channels, num_channels, stride=1, use_pre_activation=False, cardinality=32):
        """ResNeXt Basic Block. Two 3x3 convolutions with BatchNorm and activation.

        Args:
            in_channels: Number of channels in the input.
            num_channels: Number of channels in the basic block (output has num_channels * expansion).
            stride: Stride to use in the second convolutional layer.
            use_pre_activation: If True, use pre-activation ordering. Otherwise use the original formulation.
            cardinality: Number of transformations to aggregate.
        """
        self.cardinality = cardinality
        super(ResNeXtBottleneck, self).__init__(in_channels, num_channels, stride, use_pre_activation)

    def _get_conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """Aggregated convolution for ResNeXt."""
        cardinality = self.cardinality if kernel_size > 1 else 1
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=cardinality, bias=bias)

        return conv
