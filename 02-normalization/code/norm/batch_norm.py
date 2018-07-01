import torch.nn as nn


class BatchNorm2d(nn.Module):
    """Simplified 2D BatchNorm. No running stats for test inference."""
    def __init__(self, num_features, eps=1e-5):
        super(BatchNorm2d, self).__init__()

    def forward(self, x):
        raise NotImplementedError
