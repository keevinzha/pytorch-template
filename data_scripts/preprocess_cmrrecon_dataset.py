# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/28 14:25
@Auth ： keevinzha
@File ：preprocess_cmrrecon_dataset.py
@IDE ：PyCharm
"""

import numpy as np
import scipy.io as scio
import os
import nibabel as nib
import h5py
import json

from datasets import read_data
from util.recon_tools import ifft2c, fft2c
from util.utils import mkdir
from options.process_data_options import ProcessDataOptions


def crop_data_using_mask(kspace, mask, opt):
    """
    Crop kspace data to cardiac region using mask
    :param kspace: raw kspace data, shape: [height, width, coil, slice, contrast]
    :param mask: segmentation mask, shape: [height, width, slice], nii file
    :return: cropped kspace and mask
    """

    def _next_not_nan(idx, who_is_nan, count):
        next_idx = idx + 1
        if next_idx in who_is_nan:
            next_idx = _next_not_nan(next_idx, who_is_nan, count)
        who_is_nan[count] = -1
        return next_idx

    def _previous_not_nan(idx, who_is_nan, count):
        previous_idx = idx - 1
        if previous_idx in who_is_nan:
            previous_idx = _previous_not_nan(previous_idx, who_is_nan, count)
        who_is_nan[count] = -1
        return previous_idx

    for base_dir, dirs, files in os.walk(opt.original_data_root):
        if not dirs:
            for file in files:
                if opt.kspace_file_name in file:
                    data_path = os.path.join(base_dir, file)
                    base_name = os.path.basename(base_dir)
                    roi_mask_path = os.path.join(opt.roi_mask_root, base_name, opt.roi_mask_name)
                    singleslice_output_path = os.path.join(opt.singleslice_output_root, base_name)
                    multislice_output_path = os.path.join(opt.multislice_output_root, base_name)
                    mkdir(singleslice_output_path)
                    mkdir(multislice_output_path)

                    if not os.path.exists(roi_mask_path):
                        raise Exception('roi mask does not exist')
                    data = read_data(data_path, 'kspace_full')
                    mask = nib.load(roi_mask_path).get_fdata()

                    #  find the center of the heart
                    center_coordinate_list = []
                    which_slice_no_mask = []
                    for islice in range(mask.shape[-1]):
                        nx_center = np.median(np.where(mask[..., islice])[0])
                        ny_center = np.median(np.where(mask[..., islice])[1])
                        center_coordinate_list.append([nx_center, ny_center])
                        if np.isnan(nx_center):
                            which_slice_no_mask.append(islice)
                    #  use the nearest slice's center coordinate for the slices without segment mask
                    count = 0
                    for inan in which_slice_no_mask:
                        if inan == 0:
                            center_coordinate_list[inan] = center_coordinate_list[_next_not_nan(inan, which_slice_no_mask, count)]
                        else:
                            center_coordinate_list[inan] = center_coordinate_list[_previous_not_nan(inan, which_slice_no_mask, count)]
                        count += 1

                    #  crop the kspace data and mask to the cardiac region
                    islice = 0
                    for nx_center, ny_center in center_coordinate_list:  # loop over slices
                        if nx_center >= opt.cropped_data_shape[0] // 2 \
                        and ny_center >= opt.cropped_data_shape[1] // 2 \
                        and nx_center + opt.cropped_data_shape[0] // 2 <= data.shape[0] \
                        and ny_center + opt.cropped_data_shape[1] // 2 <= data.shape[1]:
                            nx_range = list(range(nx_center-opt.cropped_data_shape[0]//2, nx_center+opt.cropped_data_shape[0]//2))
                            ny_range = list(range(ny_center-opt.cropped_data_shape[1]//2, ny_center+opt.cropped_data_shape[1]//2))
                            temp = crop_kspace_data(data, nx_range, ny_range)
                            temp = np.expand_dims(temp, axis=2)
                            # todo save single slice kspace data

                            if islice == 0:
                                cropped_kspace = temp
                            else:
                                cropped_kspace = np.concatenate((cropped_kspace, temp), axis=2)
                            islice += 1
                        else:
                            raise Exception('crop out of range')
                        # todo save multi slice kspace data
                        # todo save cropped roi mask

def crop_kspace_data(kspace, nx_range, ny_range=None):
    """
    Crop kspace data
    """
    img = ifft2c(kspace)  # crop in image domain
    cropped_img = img[nx_range, ...]
    if ny_range is not None:
        cropped_img = cropped_img[:, ny_range, ...]
    cropped_kspace = fft2c(cropped_img)
    return cropped_kspace

def crop_image_data(img, nx_range, ny_range=None):
    """
    Crop kspace data
    """
    cropped_img = img[nx_range, ...]
    if ny_range is not None:
        cropped_img = cropped_img[:, ny_range, ...]
    return cropped_img

def crop_existing_data():
    """
    Crop existing data to the cardiac region
    Warning: this is a disposable function, it is not recommended to use it
    """
    ori_root = '/data/disk1/datasets/T1map_dataset/roi_cropped/roi_144'
    output_root = '/data/disk1/datasets/T1map_dataset/roi_cropped/roi_128'
    for base_dir, dirs, files in os.walk(ori_root):
        if not dirs:
            for file in files:
                if 'T1map_label.mat' in file:
                    data_path = os.path.join(base_dir, file)
                    base_name = os.path.basename(base_dir)
                    output_path = os.path.join(output_root, base_name)
                    mkdir(output_path)

                    data = read_data(data_path, 'mask')
                    nx_range = list(range(9, 9+128))
                    ny_range = list(range(9, 9+128))
                    cropped_data = crop_image_data(data, nx_range, ny_range)
                    scio.savemat(os.path.join(output_path, file), {'mask': cropped_data})

def single_slice_to_multislice(args):
    """
    Convert single slice kspace data to multislice kspace data
    """
    for base_dir, dirs, files in os.walk(args.singleslice_output_root):
        if not dirs:
            for file in sorted(files):
                if 'T1map.mat' in file:
                    data_path = os.path.join(base_dir, file)
                    base_name = os.path.basename(base_dir)
                    output_path = os.path.join(args.multislice_output_root, base_name)
                    mkdir(output_path)

                    data = read_data(data_path, 'kspace')
                    if '0' in file:
                        multislice_data = np.expand_dims(data, axis=2)
                    else:
                        multislice_data = np.concatenate((multislice_data, np.expand_dims(data, axis=2)), axis=2)
            scio.savemat(os.path.join(output_path, 'T1map.mat'), {'kspace': multislice_data})

def copy_csv(args):
    """
    Copy csv file to the new directory
    """
    for base_dir, dirs, files in os.walk(args.original_data_root):
        if not dirs:
            for file in files:
                if 'T1map.csv' in file:
                    data_path = os.path.join(base_dir, file)
                    base_name = os.path.basename(base_dir)
                    output_path = os.path.join(args.processed_data_root, base_name)
                    mkdir(output_path)

                    cmd_command = 'cp' + ' ' + data_path + ' ' + output_path
                    os.system(cmd_command)



if __name__ == '__main__':
    opt = ProcessDataOptions().parse()
    single_slice_to_multislice(opt)
    copy_csv(opt)
