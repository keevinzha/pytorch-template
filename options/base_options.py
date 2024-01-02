# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/17 10:39
@Auth ： keevinzha
@File ：base_options.py
@IDE ：PyCharm
"""

import argparse
import os

import timm
import torch

from util import utils
import models
import datasets
from models import build_model


class BaseOptions():
    """
    This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def str2bool(self, v):
        """
        Converts string to bool type; enables command line
        arguments in the format of '--arg1 true --arg2 false'
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def initialize(self, parser):
        """
        Define the common options that are used in both training and test.
        """
        # Basic parameters
        parser.add_argument('--batch_size', default=64, type=int,
                            help='Per GPU batch size')
        parser.add_argument('--epochs', default=300, type=int)
        parser.add_argument('--update_freq', default=1, type=int,
                            help='gradient accumulation steps')

        # Model parameters
        parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                            help='Name of model to train')
        parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                            help='Drop path rate (default: 0.0)')
        parser.add_argument('--input_size', default=128, type=int,
                            help='image input size')
        parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                            help="Layer scale initial values")
        parser.add_argument('--input_channels', default=9, type=int)

        # EMA related parameters
        parser.add_argument('--model_ema', type=self.str2bool, default=False)
        parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
        parser.add_argument('--model_ema_force_cpu', type=self.str2bool, default=False, help='')
        parser.add_argument('--model_ema_eval', type=self.str2bool, default=False, help='Using ema to eval during training.')

        # Augmentation parameters
        parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                            help='Color jitter factor (default: 0.4)')
        parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
        parser.add_argument('--smoothing', type=float, default=0.1,
                            help='Label smoothing (default: 0.1)')
        parser.add_argument('--train_interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')




        self.initialized = True
        return parser

    def gather_options(self):
        """
        Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options, this is a hook for model and dataset specific options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model_related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify dataset_related parser options
        dataset_name = opt.dataset
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        # save and return the parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def parse(self):
        """Parse options."""
        opt = self.gather_options()
        self.opt = opt
        return self.opt

