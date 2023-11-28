# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/28 13:30
@Auth ： keevinzha
@File ：segmask_to_bbox.py
@IDE ：PyCharm
"""

import nibabel as nib
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json


def segmask_to_bbox(segmask):
    """
    :param segmask: 3D numpy array
    :return: bbox: 3D numpy array
    """
    _, segmask = cv2.threshold(segmask, 0, 1, cv2.THRESH_BINARY)
    segmask = segmask.astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(segmask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    stats = np.squeeze(stats[:-1])
    x0, y0 = stats[0], stats[1]
    x1 = stats[0] + stats[2]
    y1 = stats[1] + stats[3]
    start_point, end_point = (x0, y0), (x1, y1)
    # draw bbox
    color = (255, 0, 0)
    thickness = 1
    mask_bboxs = cv2.rectangle(segmask, start_point, end_point, color, thickness)
    # save bbox
    save_path = ''  # todo add path when use
    dict = {'start_point': start_point, 'end_point': end_point}
    with open(save_path, 'w') as f:
        json.dump(dict, f)


def read_nii_mask(path):
    mask = nib.load(path)
    mask = mask.get_fdata()
    return mask


if __name__ == '__main__':
    path = os.path.join('/data/disk1/datasets/T1map_dataset/SegmentROI/P001', 'T1map_label.nii.gz')
    mask = read_nii_mask(path)
    bbox = segmask_to_bbox(mask[:, :, 0])
    plt.imshow(bbox)
    plt.show()