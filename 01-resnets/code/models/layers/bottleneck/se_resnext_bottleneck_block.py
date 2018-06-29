import torch.nn as nn

from .base_bottleneck_block import BaseBottleneckBlock
from models.layers import SEBlock


class SEResNeXtBottleneckBlock(BaseBottleneckBlock):

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
        super(SEResNeXtBottleneckBlock, self).__init__(in_channels, num_channels, stride, use_pre_activation)

        self.se_block = SEBlock(num_channels)

    def _get_conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """Aggregated convolution for ResNeXt."""
        cardinality = self.cardinality if kernel_size > 1 else 1
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=cardinality, bias=bias)

        return conv

    def forward(self, x):

        skip = self.skip_conn(x)

        if self.use_pre_activation:
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.conv1(x)

            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv2(x)

            x = self.norm3(x)
            x = self.relu3(x)
            x = self.conv3(x)
            x = self.se_block(x)
            x = x + skip
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = self.norm2(x)
            x = self.relu2(x)

            x = self.conv3(x)
            x = self.norm3(x)
            x = self.se_block(x)
            x = x + skip
            x = self.relu3(x)

        return x
