import torch

def set_spawn_enabled():
    """Set PyTorch start method to spawn a new process rather than spinning up a new thread.

    This change was necessary to allow multiple DataLoader workers to read from an HDF5 file.

    See Also:
        https://github.com/pytorch/pytorch/issues/3492
    """
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass


def get_batchnorm_fn(norm, relu, conv):
    """Get a function to perform BatchNorm.

    Used for efficient DenseNet implementation, where BatchNorm needs to operate on shared memory.

    Args:
        norm: Normalization layer.
        relu: ReLU layer.
        conv: Convolutional layer.

    Returns:
        Function which can be called on inputs to do BatchNorm.
    """
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function
