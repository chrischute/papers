import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from args import ArgParser
from data_loader import get_cifar_loaders
from logger import TrainLogger
from saver import ModelSaver
from tqdm import tqdm


def train(args):
    """Train model.

    Args:
        args: Command line arguments.
        model: Classifier model to train.
    """
    # Set up model
    model = models.__dict__[args.model](**vars(args))
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)

    # Set up data loader
    train_loader, test_loader, classes = get_cifar_loaders(args.batch_size, args.num_workers)

    # Set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_gamma)
    loss_fn = nn.CrossEntropyLoss().to(args.device)

    # Set up checkpoint saver
    saver = ModelSaver(model, optimizer, scheduler, args.save_dir, {'model': args.model},
                       max_to_keep=args.max_ckpts, device=args.device)

    # Train
    logger = TrainLogger(args, len(train_loader.dataset))

    while not logger.is_finished_training():
        logger.start_epoch()

        # Train for one epoch
        model.train()
        for inputs, labels in train_loader:
            logger.start_iter()

            with torch.set_grad_enabled(True):
                # Forward
                outputs = model.forward(inputs.to(args.device))
                loss = loss_fn(outputs, labels.to(args.device))
                loss_item = loss.item()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter({'loss': loss_item})

        # Evaluate on validation set
        val_loss = evaluate(model, test_loader, loss_fn, device=args.device)
        logger.write('[epoch {}]: val_loss: {:.3g}'.format(logger.epoch, val_loss))
        logger.write_summaries({'loss': val_loss}, phase='val')
        if logger.epoch in args.save_epochs:
            saver.save(logger.epoch, val_loss)

        logger.end_epoch()
        scheduler.step()


def evaluate(model, data_loader, loss_fn, device='cpu'):
    """Evaluate the model."""
    model.eval()
    losses = []
    print('Evaluating model...')
    for inputs, labels in tqdm(data_loader):
        with torch.no_grad():
            # Forward
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            losses.append(loss.item())

    return np.mean(losses)


if __name__ == '__main__':
    parser = ArgParser()
    train(parser.parse_args())
