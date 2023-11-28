# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/17 13:08
@Auth ： keevinzha
@File ：train_options.py.py
@IDE ：PyCharm
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Optimization parameters
        parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            help='Optimizer (default: "adamw"')
        parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')
        parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: None, use opt default)')
        parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                            help='Clip gradient norm (default: None, no clipping)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help='weight decay (default: 0.05)')
        parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
                        weight decay. We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs.""")

        # Learning rate schedule parameters
        parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                            help='learning rate (default: 4e-3), with total batch size 4096')
        parser.add_argument('--layer_decay', type=float, default=1.0)
        parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
        parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
        parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                            help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

        # * Random Erase params
        parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
        parser.add_argument('--remode', type=str, default='pixel',
                            help='Random erase mode (default: "pixel")')
        parser.add_argument('--recount', type=int, default=1,
                            help='Random erase count (default: 1)')
        parser.add_argument('--resplit', type=self.str2bool, default=False,
                            help='Do not random erase first (clean) augmentation split')

        # * Mixup params
        parser.add_argument('--mixup', type=float, default=0,
                            help='mixup alpha, mixup enabled if > 0.')
        parser.add_argument('--cutmix', type=float, default=0,
                            help='cutmix alpha, cutmix enabled if > 0.')
        parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                            help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
        parser.add_argument('--mixup_prob', type=float, default=1.0,
                            help='Probability of performing mixup or cutmix when either/both is enabled')
        parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                            help='Probability of switching to cutmix when both mixup and cutmix enabled')
        parser.add_argument('--mixup_mode', type=str, default='batch',
                            help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

        # * Finetuning params
        parser.add_argument('--finetune', default='',
                            help='finetune from checkpoint')
        parser.add_argument('--head_init_scale', default=1.0, type=float,
                            help='classifier head initial scale, typically adjusted in fine-tuning')
        parser.add_argument('--model_key', default='model|module', type=str,
                            help='which key to load from saved state dict, usually model or model_ema')
        parser.add_argument('--model_prefix', default='', type=str)

        # Dataset parameters
        parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                            help='dataset path')
        parser.add_argument('--eval_data_path', default=None, type=str,
                            help='dataset path for evaluation')
        parser.add_argument('--nb_classes', default=1000, type=int,
                            help='number of the classification types')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"))
        parser.add_argument('--imagenet_default_mean_and_std', type=self.str2bool, default=True)
        parser.add_argument('--dataset', default='base',
                            type=str, help='prefix of your dataset.py')
        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--log_dir', default=None,
                            help='path where to tensorboard log')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=0, type=int)

        parser.add_argument('--resume', default='',
                            help='resume from checkpoint')
        parser.add_argument('--auto_resume', type=self.str2bool, default=True)
        parser.add_argument('--save_ckpt', type=self.str2bool, default=True)
        parser.add_argument('--save_ckpt_freq', default=1, type=int)
        parser.add_argument('--save_ckpt_num', default=3, type=int)

        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', type=self.str2bool, default=False,
                            help='Perform evaluation only')
        parser.add_argument('--dist_eval', type=self.str2bool, default=True,
                            help='Enabling distributed evaluation')
        parser.add_argument('--disable_eval', type=self.str2bool, default=False,
                            help='Disabling evaluation during training')
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--pin_mem', type=self.str2bool, default=True,
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--local-rank', default=-1, type=int)
        parser.add_argument('--dist_on_itp', type=self.str2bool, default=False)
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')

        parser.add_argument('--use_amp', type=self.str2bool, default=False,
                            help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

        # Weights and Biases arguments
        parser.add_argument('--enable_wandb', type=self.str2bool, default=False,
                            help="enable logging to Weights and Biases")
        parser.add_argument('--project', default='convnext', type=str,
                            help="The name of the W&B project where you're sending the new run.")
        parser.add_argument('--wandb_ckpt', type=self.str2bool, default=False,
                            help="Save model checkpoints as W&B Artifacts.")

        self.isTrain = True
        return parser