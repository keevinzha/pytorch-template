# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/28 14:32
@Auth ： keevinzha
@File ：process_data_options.py
@IDE ：PyCharm
"""
import argparse

from . base_options import BaseOptions

class ProcessDataOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--original_data_root', type=str, default='', help='path to original data')
        parser.add_argument('--processed_data_root', type=str, default='', help='path to processed data')
        parser.add_argument('--roi_mask_root', type=str, default='', help='path to roi mask')
        parser.add_argument('--roi_mask_name', type=str, default='T1map_label.nii.gz', help='name of roi mask')
        parser.add_argument('--kspace_file_name', type=str, default='T1map.mat', help='name of kspace file')
        parser.add_argument('--cropped_data_shape', type=list, default=[128, 128], help='cropped data shape')
        parser.add_argument('--singleslice_output_root', type=str, default='', help='path to singleslice data')
        parser.add_argument('--multislice_output_root', type=str, default='', help='path to multislice data')
        parser.add_argument('--cropped_mask_output_root', type=str, default='', help='path to cropped mask')
        parser.add_argument('--bbox_root', type=str, default='', help='path to bbox')
        parser.add_argument('--jpg_root', type=str, default='/data/disk1/datasets/T1map_dataset/pictures/image', help='path to jpg')
        parser.add_argument('--yolo_image_root', type=str, default='', help='path for yolo dataset')
        parser.add_argument('--yolo_label_root', type=str, default='', help='path for yolo dataset')
        return parser

    def parse(self):
        parser = self.initialize()
        opt = parser.parse_args()
        return opt

