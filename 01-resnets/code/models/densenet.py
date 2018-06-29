import torch.nn as nn

from models.layers import DenseNetBottleneck, DenseNetTransition, GAPLayer


class DenseNet(nn.Module):
    """DenseNet for CIFAR-10.

    Based on the paper:
    "Densely Connected Convolutional Networks"
    by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    (https://arxiv.org/abs/1608.06993).
    """

    depth2config = {58: [6, 12, 24, 16]}  # CIFAR-10

    def __init__(self, model_depth, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.num_channels = 2 * growth_rate

        # Get number of blocks per layer
        try:
            block_cfg = self.depth2config[model_depth]
        except KeyError:
            raise ValueError('Unsupported model depth for ResNet: {} (supports {})'
                             .format(model_depth, self.depth2config.keys()))

        # Input conv (no max pool for CIFAR-10)
        self.in_conv = nn.Sequential(nn.Conv2d(3, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(self.num_channels))

        # Layers of residual blocks
        dense_layers = []
        for i, num_blocks in enumerate(block_cfg):
            dense_layers.append(self._get_layer(self.num_channels, num_blocks))

            self.num_channels += num_blocks * self.growth_rate

            if i < len(block_cfg) - 1:
                # Add a transition block after the dense block
                out_channels = int(self.num_channels * reduction // 1)
                dense_layers.append =(DenseNetTransition(self.num_channels, out_channels))
                self.num_channels = out_channels

        self.dense_layers = nn.Sequential(*dense_layers)

        # Global average pool
        self.classifier = GAPLayer(self.in_channels, num_classes)

    def _get_layer(self, in_channels, num_blocks):
        dense_blocks = [DenseNetBottleneck(in_channels * self.growth_rate**i, self.growth_rate)
                        for i in range(num_blocks)]
        dense_layer = nn.Sequential(*dense_blocks)

        return dense_layer

    def forward(self, x):
        x = self.in_conv(x)
        x = self.dense_layers(x)
        x = self.classifier(x)

        return x
