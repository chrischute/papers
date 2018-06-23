import os
import torch.nn.functional as F

from time import time
from datetime import datetime
from tensorboardX import SummaryWriter


class TestLogger(object):
    """Class for logging test info to the console and saving test outputs to disk."""
    def __init__(self, args):
        super(TestLogger, self).__init__()
        self.epoch_start_time = None

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'log.txt')

        log_dir = os.path.join(args.log_dir, '_'.join([args.name, datetime.now().strftime('%b%d_%H%M'), 'val']))
        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.y_true_buckets = []
        self.y_pred_buckets = []
        self.num_buckets = 4

    def start_iter(self):
        """Log info for start of an iteration."""
        pass

    def end_iter(self, filters, targets, logits):
        """Log info for end of an iteration."""
        probs = F.sigmoid(logits.detach())
        tgts = targets.detach()

        batch_size = filters.size(0)
        for i in range(batch_size):
            t = tgts[i].item()
            p = probs[i].item()
            f = filters[i]

            # Normalize filter values into 0, 1
            f -= f.min()
            f /= f.max()

            # Move channels last
            f.transpose(0, 1)
            f.transpose(1, 2)

            self.summary_writer.add_image('actual_{:.1f}/pred_{:.1f}'.format(t, p), f)

            self.y_pred_buckets.append(p * self.num_buckets // 1)
            self.y_true_buckets.append(t * self.num_buckets // 1)

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.write('[start of test: writing to {}]'.format(self.save_dir))

    def end_epoch(self):
        """Log info for end of an epoch."""
        self.write('[end of test, time: {:.2g}]'.format(time() - self.epoch_start_time))
        self.summary_writer.file_writer.flush()

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)
