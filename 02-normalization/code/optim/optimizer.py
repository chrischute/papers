import torch.optim as optim


def get_optimizer(parameters, args):
    """Get a PyTorch optimizer for params.

    Args:
        parameters: Iterator of network parameters to optimize (i.e., model.parameters()).
        args: Command line arguments.

    Returns:
        PyTorch optimizer specified by args_.
    """
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, args.lr,
                              momentum=args.sgd_momentum,
                              nesterov=True,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(parameters, args.lr,
                               betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(args.optimizer))

    return optimizer
