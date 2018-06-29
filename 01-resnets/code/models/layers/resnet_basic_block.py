import torch.nn as nn


class ResNetBasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, num_channels, stride=1, use_pre_activation=False):
        """ResNet Basic Block. Two 3x3 convolutions with BatchNorm and activation.

        Args:
            in_channels: Number of channels in the input.
            num_channels: Number of channels in the basic block (output has num_channels * expansion).
            stride: Stride to use in the first convolutional layer.
            use_pre_activation: If True, use pre-activation ordering. Otherwise use the original formulation.
        """
        super(ResNetBasicBlock, self).__init__()

        self.use_pre_activation = use_pre_activation
        out_channels = self.expansion * num_channels

        skip_layers = []
        if stride != 1 or in_channels != out_channels:
            skip_layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels)]
        self.skip_conn = nn.Sequential(*skip_layers)

        # No bias in Conv2d because the BatchNorm layer has a bias
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels if use_pre_activation else num_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_channels if use_pre_activation else out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        skip = self.skip_conn(x)

        if self.use_pre_activation:
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.conv1(x)

            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv2(x)
            x = x + skip
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = self.norm2(x)
            x = x + skip
            x = self.relu2(x)

        return x
