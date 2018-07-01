from evaluator import AverageMeter
from time import time
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""
    def __init__(self, args, dataset_len):
        super(TrainLogger, self).__init__(args, dataset_len)

        assert args.is_training, 'TrainLogger should only be used during training.'

        self.iters_per_print = args.iters_per_print
        self.iters_until_print = args.iters_per_print

        self.iters_per_visual = args.iters_per_visual
        self.iters_until_visual = args.iters_per_visual

        self.experiment_name = args.name
        self.max_eval = args.max_eval
        self.num_epochs = args.num_epochs
        self.loss_meter = AverageMeter()

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def log_iter(self, loss):
        """Log results from a training iteration."""
        loss = loss.item()
        self.loss_meter.update(loss, self.batch_size)

        # Periodically write to the log and TensorBoard
        if self.iter % self.iters_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}, loss: {:.3g}]' \
                .format(self.epoch, self.iter, self.dataset_len, avg_time, self.loss_meter.avg)

            # Write all errors as scalars to the graph
            self._log_scalars({'batch_loss': self.loss_meter.avg}, print_to_stdout=False)
            self.loss_meter.reset()

            self.write(message)

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))

    def end_epoch(self, metrics):
        """Log info for end of an epoch.

        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
        """
        self.write('[end of epoch {}, epoch time: {:.2g}]'.format(self.epoch, time() - self.epoch_start_time))
        self._log_scalars(metrics)

        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
