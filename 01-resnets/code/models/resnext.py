from models.base_resnet import BaseResNet
from models.layers import ResNeXtBottleneck


class ResNeXt(BaseResNet):
    """ResNeXt for CIFAR10.

    Based on the paper:
    "Aggregated Residual Transformations for Deep Neural Networks"
    by Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    (https://arxiv.org/abs/1611.05431).
    """

    depth2block = {29: ResNeXtBottleneck}

    depth2config = {29: [3, 3, 3]}

    def __init__(self, model_depth, init_channels, num_classes, **kwargs):
        """
        Args:
            model_depth: Depth of the ResNet model.
            init_channels: Number of channels in the initial residual block.
            num_classes: Number of classes on the output head.
        """
        super(ResNeXt, self).__init__(model_depth, init_channels, num_classes, use_pre_activation=False)
