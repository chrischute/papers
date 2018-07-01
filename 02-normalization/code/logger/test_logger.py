from time import time
from .base_logger import BaseLogger


class TestLogger(BaseLogger):
    """Class for logging test info to the console and saving test outputs to disk."""
    def __init__(self, args, dataset_len):
        assert not args.is_training
        super(TestLogger, self).__init__(args, dataset_len)

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of test: writing to {}]'.format(self.save_dir))

    def end_epoch(self, metrics):
        """Log info for end of an epoch.

        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
        """
        self._log_scalars(metrics)
        self.write('[end of test, time: {:.2g}]'.format(time() - self.epoch_start_time))
