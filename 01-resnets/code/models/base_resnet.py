import torch.nn as nn

from models.layers import GAPLayer


class BaseResNet(nn.Module):
    """Base class for ResNets for CIFAR-10.

    Based on the paper:
    "Deep Residual Learning for Image Recognition"
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    (https://arxiv.org/abs/1512.03385).
    """

    def __init__(self, model_depth, init_channels, num_classes, use_pre_activation=True, **kwargs):
        """

        Args:
            model_depth: Depth of the ResNet model.
            init_channels: Number of channels in the initial residual block.
            num_classes: Number of classes on the output head.
            use_pre_activation: If true, use pre-activation order in the residual blocks.
        """
        super(BaseResNet, self).__init__()

        self.in_channels = init_channels
        self.use_pre_activation = use_pre_activation

        try:
            block_fn = self.depth2block[model_depth]
            block_cfg = self.depth2config[model_depth]
        except KeyError:
            raise ValueError('Unsupported model depth for ResNet: {}'.format(model_depth))

        # Input conv (no max pool for CIFAR-10)
        self.in_conv = nn.Sequential(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(self.in_channels))

        # Layers of residual blocks
        res_layers = [self._get_layer(block_fn, init_channels * 2**i, num_blocks, stride=min(i+1, 2))
                      for i, num_blocks in enumerate(block_cfg)]
        self.res_layers = nn.Sequential(*res_layers)

        # Global average pool
        self.classifier = GAPLayer(self.in_channels, num_classes)

    def _get_layer(self, block_fn, num_channels, num_blocks, stride):
        """Get a layer of residual blocks.

        Notes:
            Changes `self.in_channels`.

        Args:
            block_fn: Constructor for type of block to use.
            num_channels: Number of channels in the first residual block.
            num_blocks: Number of residual blocks in the layer.
            stride: Stride of for the first residual block.

        Returns:
            Callable layer of residual blocks.
        """
        res_blocks = [block_fn(self.in_channels, num_channels, stride, self.use_pre_activation)]
        self.in_channels = num_channels * block_fn.expansion
        res_blocks += [block_fn(self.in_channels, num_channels, use_pre_activation=self.use_pre_activation)
                       for _ in range(num_blocks - 1)]
        res_layer = nn.Sequential(*res_blocks)

        return res_layer

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_layers(x)
        x = self.classifier(x)

        return x
