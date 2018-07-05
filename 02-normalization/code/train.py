import models
import optim
import torch
import torch.nn as nn

from args import TrainArgParser
from data_loader import CIFARLoader
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver


def train(args):

    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    lr_scheduler = optim.get_scheduler(optimizer, args)
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)

    # Get logger, evaluator, saver
    loss_fn = nn.CrossEntropyLoss()
    train_loader = CIFARLoader('train', args.batch_size, args.num_workers)
    logger = TrainLogger(args, len(train_loader.dataset))
    eval_loaders = [CIFARLoader('train', args.batch_size, args.num_workers),
                    CIFARLoader('val', args.batch_size, args.num_workers)]
    evaluator = ModelEvaluator(eval_loaders, logger, args.max_eval, args.epochs_per_eval)
    saver = ModelSaver(**vars(args))

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for inputs, targets in train_loader:
            logger.start_iter()
            
            with torch.set_grad_enabled(True):
                logits = model.forward(inputs.to(args.device))
                loss = loss_fn(logits, targets.to(args.device))

                logger.log_iter(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter()

        metrics = evaluator.evaluate(model, args.device, logger.epoch)
        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
                   metric_val=metrics.get(args.metric_name, None))
        logger.end_epoch(metrics)
        optim.step_scheduler(lr_scheduler, metrics, logger.epoch)


if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())
