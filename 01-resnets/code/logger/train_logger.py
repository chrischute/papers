import os

from datetime import datetime
from tensorboardX import SummaryWriter
from time import time


class TrainLogger(object):
    """Class for logging training info to the console."""

    def __init__(self, args, dataset_len):
        super(TrainLogger, self).__init__()

        assert args.iters_per_print % args.batch_size == 0, "iters_per_print must be divisible by batch_size"
        assert args.iters_per_visual % args.batch_size == 0, "iters_per_visual must be divisible by batch_size"

        self.iters_per_print = args.iters_per_print
        self.iters_per_visual = args.iters_per_visual
        self.num_visuals = args.num_visuals
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.dataset_len = dataset_len

        self.epoch = 1
        self.global_step = 0
        self.iter = 0
        self.iter_start_time = None
        self.epoch_start_time = None
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'log.txt')
        # Use two summary writers to plot train/val loss on the same graph
        self.summary_writers = {}
        for phase in ('train', 'val'):
            log_dir = os.path.join(args.log_dir, '_'.join([args.name, datetime.now().strftime('%b%d_%H%M'), phase]))
            self.summary_writers[phase] = SummaryWriter(log_dir=log_dir)

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def end_iter(self, loss_dict):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

        # Periodically write to the log and TensorBoard
        if self.global_step % self.iters_per_print == 0:
            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}] ' \
                .format(self.epoch, self.iter, self.dataset_len, avg_time)

            # Write the current error report
            loss_strings = ['{}: {:.3g}'.format(k, loss_dict[k]) for k in loss_dict]
            message += ', '.join(loss_strings)

            # Write all errors as scalars to the graph
            self.write_summaries(loss_dict, phase='train')

            self.write(message)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.global_step % self.iters_per_visual == 0:
            for i in range(self.num_visuals):
                # TODO: Log an image to tensorboard
                pass

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))

    def end_epoch(self):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        self.write('[end of epoch {}, epoch time: {:.2f}]'.format(self.epoch, time() - self.epoch_start_time))
        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def write_summaries(self, loss_dict, phase='train'):
        """Log summaries to TensorBoard. Underscore-separated tokens become forward-slash-separated."""
        for k, v in loss_dict.items():
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writers[phase].add_scalar(k, v, self.global_step)
