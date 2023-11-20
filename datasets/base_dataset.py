# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/16 16:12
@Auth ： keevinzha
@File ：base_dataset.py
@IDE ：PyCharm
"""

import random
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BaseDataset(Dataset, ABC):
    """
    This code comes from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a datasets point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, is_train):
        """
        Initialize the class; save the options in the class
        :param opt(Option class): stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.data_path if is_train else opt.eval_data_path

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add new dataset-specific options, and rewrite default values for existing options.
        :param parser: original option parser
        :param is_train(bool): whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """
        Return the total number of images in the dataset.
        :return: int
        """
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """
        Return a datasets point and its metadata information.
        :param index: a random integer for datasets indexing
        :return: a dictionary of datasets with their names. It ususally contains the datasets itself and its metadata information.
        """
        pass

    # todo some pre-processing functions may be added here
def get_transform(opt):
    pass