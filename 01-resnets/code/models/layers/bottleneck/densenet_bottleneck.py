import torch
import torch.nn as nn


class DenseNetBottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseNetBottleneck, self).__init__()

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        skip = x

        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = torch.cat((x, skip), dim=1)

        return x
