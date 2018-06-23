import models
import torch.utils.data
import torch.nn as nn

from args import ArgParser
from data_loader import get_cifar_loaders
from logger import TestLogger
from saver import ModelSaver


def test(args):

    model_fn = models.__dict__[args_.model]
    model = model_fn(args.num_classes)
    model = nn.DataParallel(model, args.gpu_ids)

    ckpt_info = ModelSaver.load_model(args.ckpt_path, model)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    _, test_loader, _ = get_cifar_loaders(args.batch_size, args.num_workers)
    logger = TestLogger(args)

    logger.start_epoch()
    for inputs, labels in test_loader:
        logger.start_iter()

        with torch.set_grad_enabled(True):
            # Forward
            logits = model.forward(inputs.to(args.device))

        logger.end_iter(inputs, labels, logits)
    logger.end_epoch()


if __name__ == '__main__':
    parser = ArgParser()
    args_ = parser.parse_args()
    test(args_)
