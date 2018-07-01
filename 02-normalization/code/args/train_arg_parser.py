from .base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        # Logger args
        self.parser.add_argument('--epochs_per_save', type=int, default=5,
                                 help='Number of epochs between saving a checkpoint to save_dir.')
        self.parser.add_argument('--iters_per_print', type=int, default=4,
                                 help='Number of iterations between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--epochs_per_eval', type=int, default=1,
                                 help='Number of epochs between evaluating model on the validation set.')
        self.parser.add_argument('--iters_per_visual', type=int, default=80,
                                 help='Number of iterations between visualizing training examples.')
        self.parser.add_argument('--num_epochs', type=int, default=300,
                                 help='Number of epochs to train. If 0, train forever.')

        # Evaluator args
        self.parser.add_argument('--max_ckpts', type=int, default=2,
                                 help='Number of recent ckpts to keep before overwriting old ones.')
        self.parser.add_argument('--max_eval', type=int, default=-1,
                                 help='Max number of examples to evaluate from the training set.')
        self.parser.add_argument('--metric_name', type=str, default='val_loss', choices=('val_loss', 'val_AUROC'),
                                 help='Metric used to determine which checkpoint is best.')

        # Optimizer args
        self.parser.add_argument('--adam_beta_1', type=float, default=0.9, help='Adam beta 1 (Adam only).')
        self.parser.add_argument('--adam_beta_2', type=float, default=0.999, help='Adam beta 2 (Adam only).')
        self.parser.add_argument('--lr', type=float, default=1e-1, help='Initial learning rate.')
        self.parser.add_argument('--lr_scheduler', type=str, default='step', choices=('step', 'multi_step', 'plateau'),
                                 help='LR scheduler to use.')
        self.parser.add_argument('--lr_decay_gamma', type=float, default=0.1,
                                 help='Multiply learning rate by this value every LR step (step and multi_step only).')
        self.parser.add_argument('--lr_decay_step', type=int, default=100,
                                 help='Number of epochs between each multiply-by-gamma step.')
        self.parser.add_argument('--lr_milestones', type=str, default='50,125,250',
                                 help='Epochs to step the LR when using multi_step LR scheduler.')
        self.parser.add_argument('--lr_patience', type=int, default=10,
                                 help='Number of stagnant epochs before stepping LR.')
        self.parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adam'), help='Optimizer.')
        self.parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum (SGD only).')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4,
                                 help='Weight decay (i.e., L2 regularization factor).')
