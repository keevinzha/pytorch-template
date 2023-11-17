# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/17 10:39
@Auth ： keevinzha
@File ：base_options.py
@IDE ：PyCharm
"""

import argparse
import os

import torch

from util import util
import models
import data


class BaseOptions():
    """
    This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        """
        Define the common options that are used in both training and test.
        """
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to dataset')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='unet', help='chooses which model to use. ')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input data channels: usually time frame or number of contrasts, or 2 for read and imaginary part')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output data channels')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]') # todo complete this
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='mapping', help='chooses how datasets are loaded. [mapping]')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes data in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')  # todo modify this
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')


