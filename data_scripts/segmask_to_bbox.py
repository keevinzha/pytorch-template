# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/28 13:30
@Auth ： keevinzha
@File ：segmask_to_bbox.py
@IDE ：PyCharm
"""
from math import floor

import nibabel as nib
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
import scipy.io as scio

from options.process_data_options import ProcessDataOptions
from util.utils import mkdir


def segmask_to_bbox(segmask):
    """
    :param segmask: 3D numpy array
    :return: bbox: 3D numpy array
    """
    # if segmask is empty, retyrn (-1, -1), (-1, -1)
    if np.max(segmask) == 0:
        return (-1, -1), (-1, -1)
    _, segmask = cv2.threshold(segmask, 0, 1, cv2.THRESH_BINARY)
    segmask = segmask.astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(segmask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    stats = np.squeeze(stats[:-1])
    if stats.ndim != 1:
        for i in range(stats.shape[0]):
            if stats[i][4] < 5:
                continue
            else:
                stats = np.squeeze(stats[i, :])
                break
    x0, y0 = stats[0], stats[1]
    x1 = stats[0] + stats[2]
    y1 = stats[1] + stats[3]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    start_point, end_point = (x0, y0), (x1, y1)
    # draw bbox
    color = (255, 0, 0)
    thickness = 1
    mask_bboxs = cv2.rectangle(segmask, start_point, end_point, color, thickness)
    return start_point, end_point


def read_nii_mask(path):
    mask = nib.load(path)
    mask = mask.get_fdata()
    return mask

def generate_bbox_json(opt):
    for base_dir, dirs, files in os.walk(opt.roi_mask_root):
        if not dirs:
            for file in files:
                if opt.roi_mask_name in file:
                    base_name = os.path.basename(base_dir)
                    roi_mask_path = os.path.join(opt.roi_mask_root, base_name, opt.roi_mask_name)
                    bbox_output_path = os.path.join(opt.bbox_root, base_name)
                    mkdir(bbox_output_path)

                    if not os.path.exists(roi_mask_path):
                        raise Exception('roi mask does not exist')
                    mask = scio.loadmat(roi_mask_path)
                    mask = mask['mask'][:]

                    for islice in range(mask.shape[-1]):
                        start_point, end_point = segmask_to_bbox(mask[:, :, islice])
                        if start_point == (-1, -1) and end_point == (-1, -1):
                            continue
                        for iti in range(9):  # every data have 9 tis, according to 9 images
                            bbox_path = os.path.join(bbox_output_path, str(islice) + '_T1map_' + str(iti) + '.txt')
                            x_center = (start_point[0] + end_point[0]) / 2
                            y_center = (start_point[1] + end_point[1]) / 2
                            width = end_point[0] - start_point[0]
                            height = end_point[1] - start_point[1]
                            # Box coordinates must be in normalized xywh format (from 0 to 1). If your boxes are in
                            # pixels, you should divide x_center and width by image width, and y_center and height by
                            # image height.
                            x_center = x_center / mask.shape[0]
                            y_center = y_center / mask.shape[1]
                            width = width / mask.shape[0]
                            height = height / mask.shape[1]
                            if x_center < 0 or y_center < 0:
                                continue
                            # save bbox
                            line = "{} {} {} {} {}".format(0, x_center, y_center, width, height)
                            with open(bbox_path, 'w') as f:
                                f.writelines(line)



if __name__ == '__main__':
    opt = ProcessDataOptions().parse()
    generate_bbox_json(opt)