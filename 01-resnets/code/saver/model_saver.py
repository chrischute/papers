import os
import shutil
import torch


class ModelSaver(object):
    """Class to save and load model checkpoints."""
    def __init__(self, model, optimizer, lr_scheduler, save_dir, ckpt_info, max_to_keep, device):
        """
        Args:
            model: Model to save.
            optimizer: Optimizer for model parameters.
            lr_scheduler: Learning rate scheduler for optimizer.
            save_dir: Directory to save checkpoints.
            ckpt_info: Dictionary of model info to save with each checkpoint, return with load.
            max_to_keep: Number of checkpoints to keep around. If <= 0, no limit.
            device: Device where the model/optimizer parameters belong.
        """
        super(ModelSaver, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.save_dir = save_dir
        self.ckpt_info = ckpt_info
        self.max_to_keep = None if max_to_keep <= 0 else max_to_keep
        self.device = device
        self.best_val_loss = float('inf')
        self.ckpt_paths = []

    def save(self, epoch, val_loss):
        """Save model parameters to disk."""
        checkpoint_dict = {
            'epoch': epoch,
            'model': self.model.to('cpu').state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'ckpt_info': self.ckpt_info,
            'val_loss': val_loss
        }
        self.model.to(self.device)

        ckpt_path = os.path.join(self.save_dir, 'epoch_{}.pth.tar'.format(epoch))
        torch.save(checkpoint_dict, ckpt_path)

        if val_loss < self.best_val_loss:
            # Save the best model
            self.best_val_loss = val_loss
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(ckpt_path, best_path)

        # Remove a checkpoint if more than max_to_keep checkpoints saved
        self.ckpt_paths.append(ckpt_path)
        if self.max_to_keep is not None and len(self.ckpt_paths) > self.max_to_keep:
            oldest_ckpt = self.ckpt_paths.pop(0)
            os.remove(oldest_ckpt)

    @classmethod
    def load_model(cls, checkpoint_path, model):
        """Load model parameters from disk.
        Args:
            checkpoint_path: Path to checkpoint to load.
            model: Model to initialize with parameters from the checkpoint.
        Returns:
            Dictionary containing keys from ckpt_info of the `ModelSaver` that created
            the checkpoint. Returned dict also contains an `epoch` key.
        """
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        # Load epoch and ckpt_info saved with the checkpoint
        ckpt_info = {'epoch': checkpoint_dict['epoch'],
                     'val_loss': checkpoint_dict['val_loss']}
        ckpt_info.update(checkpoint_dict['ckpt_info'])

        # Load model and move parameters to device
        model.load_state_dict(checkpoint_dict['model'])

        return ckpt_info

    @classmethod
    def load_optimizer(cls, checkpoint_path, optimizer, lr_scheduler=None):
        """Load optimizer and LR scheduler state from disk
        Args:
            checkpoint_path: Path to checkpoint to load.
            optimizer: Optimizer to initialize with parameters from the checkpoint.
            lr_scheduler: Optional learning rate scheduler to initialize with parameters from the checkpoint.
        """
        checkpoint_dict = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
