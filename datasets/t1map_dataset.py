# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/20 10:15
@Auth ： keevinzha
@File ：t1map_dataset.py
@IDE ：PyCharm
"""
import os
import numpy as np
from ultralytics import YOLO

from datasets.base_dataset import BaseDataset
from datasets import build_datafile_list, read_data
from util.recon_tools import ifft2c, sos
from util.utils import yolo_detect_cardic

class T1mapDataset(BaseDataset):
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
        self.t1map_paths = [os.path.join(self.dir_t1map, os.path.basename(os.path.dirname(path)), os.path.basename(path).split('_')[1]) for path in self.kspace_paths]

    def __getitem__(self, item):
        """
        Return a datasets point and its metadata information.
        :param index: a random integer for datasets indexing
        :return: a dictionary of datasets with their names. It ususally contains the datasets itself and its metadata information.


        t1 = (1 / b) * (a / (a + c) - 1)
        T1ReconMap[:, :, slc, 0] = t1
        T1ReconMap[:, :, slc, 1] = a
        T1ReconMap[:, :, slc, 2] = b
        T1ReconMap[:, :, slc, 3] = c
        T1ReconMap[:, :, slc, 4] = filp
        """
        kspace = read_data(self.kspace_paths[item], 'kspace')
        mask = read_data(self.mask_paths[item], 'mask')
        t1map = read_data(self.t1map_paths[item], 'T1map')
        cardiac_mask = yolo_detect_cardic(self.kspace_paths[item])
        sample = kspace

        # kspace 2 image
        sample = ifft2c(sample, (0, 1))
        sample = np.squeeze(sos(sample, 2))
        # yolo detect cardiac area
        cardiac_mask = np.expand_dims(cardiac_mask, 0)

        sample = np.transpose(sample, (2, 0, 1))
        t1map_slice = os.path.basename(self.kspace_paths[item]).split('_')[0]
        t1map_slice = int(t1map_slice)
        target = np.squeeze(t1map[:, :, t1map_slice, 0])
        target = self._threshold(target, [0, 3000])
        target = np.expand_dims(target, 0)
        # undersample mask, not used
        mask = np.expand_dims(mask, 0)

        # multiply cardiac mask
        sample = cardiac_mask * sample
        target = cardiac_mask * target

        sample = sample.astype(np.float32)
        target = target.astype(np.float32)
        # trun nan to 0
        sample = np.nan_to_num(sample)
        target = np.nan_to_num(target)
        return sample, target, mask

    def __len__(self):
        """
        Return the total number of datas in the dataset.
        :return: int
        """
        return len(self.kspace_paths)

    def _threshold(self, img, threshold=[0, 3000]):
        img[img < threshold[0]] = threshold[0]
        img[img > threshold[1]] = threshold[1]
        return img

