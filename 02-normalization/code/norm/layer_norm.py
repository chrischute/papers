import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()

    def forward(self, x):
        raise NotImplementedError
