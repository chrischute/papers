from models.base_resnet import BaseResNet

from models.layers import SEResNeXtBottleneck


class SEResNeXt(BaseResNet):
    """SE-ResNeXt for CIFAR10.

    Based on the paper:
    "Squeeze-and-Excitation Networks"
    by Jie Hu, Li Shen, Gang Sun
    (https://arxiv.org/abs/1709.01507).
    """
    depth2block = {29: SEResNeXtBottleneck}

    depth2config = {29: [3, 3, 3]}

    def __init__(self, model_depth, init_channels, num_classes, **kwargs):
        """
        Args:
            model_depth: Depth of the ResNet model.
            init_channels: Number of channels in the initial residual block.
            num_classes: Number of classes on the output head.
        """
        super(SEResNeXt, self).__init__(model_depth, init_channels, num_classes, use_pre_activation=False)
