import torch.nn as nn


class DenseNetTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseNetTransition, self).__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)

        return x
