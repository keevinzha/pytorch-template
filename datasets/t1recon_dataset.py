# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/17 16:01
@Auth ： keevinzha
@File ：t1recon_dataset.py
@IDE ：PyCharm
"""
import os

import numpy as np

from datasets.base_dataset import BaseDataset
from datasets import build_datafile_list, read_data

class T1ReconDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--kspace_path', type=str, default='kspace', help='name of kspace data folder')
        parser.add_argument('--mask_path', type=str, default='mask_t1map_af8.mat', help='name of mask')
        parser.add_argument('--t1map_path', type=str, default='t1map_cropped_singleslice_trf_p0', help='name of t1map data folder')
        return parser

    def __init__(self, opt, is_train):
        """
        Initialize this dataset class
        :param opt (Option class): stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt, is_train)
        data_root = self.root
        # get datasets (kspace, mask, t1map) paths of dataset
        self.dir_kspace = os.path.join(data_root, opt.kspace_path)
        self.dir_mask = os.path.join(data_root, opt.mask_path)
        self.dir_t1map = os.path.join(data_root, opt.t1map_path)
        self.kspace_paths = sorted(build_datafile_list(self.dir_kspace, opt.max_dataset_size))
        self.mask_paths = [self.dir_mask] * len(self.kspace_paths)
        self.t1map_paths = sorted(build_datafile_list(self.dir_t1map, opt.max_dataset_size))


    def __getitem__(self, index):
        """
        Return a datasets point and its metadata information.
        :param index: a random integer for datasets indexing
        :return: a dictionary of datasets with their names. It ususally contains the datasets itself and its metadata information.
        """
        kspace = read_data(self.kspace_paths[index], 'kspace')
        mask = read_data(self.mask_paths[index], 'mask')
        t1map = read_data(self.t1map_paths[index], 'T1map')
        sample = kspace * np.expand_dims(mask, 2)
        target = kspace
        sample = np.transpose(sample, (2, 0, 1, 3))
        target = np.transpose(target, (2, 0, 1, 3))
        mask = np.expand_dims(mask, 0)
        return sample, target, mask

    def __len__(self):
        """
        Return the total number of datas in the dataset.
        :return: int
        """
        return len(self.kspace_paths)

