# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/5 13:00
@Auth ： keevinzha
@File ：mat_to_jpg.py
@IDE ：PyCharm
"""

import os
import scipy.io as scio
import numpy as np
import cv2

from options.process_data_options import ProcessDataOptions
from datasets import read_data
from util.recon_tools import ifft2c, sos
from util.utils import mkdir


def mat_to_jpg(mat_path, jpg_path):
    """
    :param mat_path: path of mat file
    :param jpg_path: path of jpg file
    :return:
    """
    data = read_data(mat_path, 'kspace')
    original_jpg_path = jpg_path
    for i in range(data.shape[3]):
        img = ifft2c(data[:, :, :, i], (0, 1))
        img = sos(img, 2)
        img = img.astype(np.float32)
        img = img / np.max(img) * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        jpg_path = original_jpg_path.replace('.jpg', '_{}.jpg'.format(i))
        cv2.imwrite(jpg_path, img)


def walk_through_root(mat_root, jpg_root, bbox_root):
    # skip file which not in bbox_root
    for base_dir, dirs, files in os.walk(mat_root):
        if not dirs:
            for file in files:
                if '.mat' in file:
                    if not os.path.exists(os.path.join(bbox_root, os.path.basename(base_dir),
                                                       file.replace('.mat', '.txt').replace('T1map', 'bbox'))):
                        continue
                    mat_path = os.path.join(base_dir, file)
                    jpg_path = mat_path.replace(mat_root, jpg_root).replace('.mat', '.jpg')
                    folder_path = os.path.dirname(jpg_path)
                    mkdir(folder_path)
                    mat_to_jpg(mat_path, jpg_path)


if __name__ == '__main__':
    opt = ProcessDataOptions().parse()
    walk_through_root(opt.processed_data_root, opt.jpg_root, opt.bbox_root)
