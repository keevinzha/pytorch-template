# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/16 16:11
@Auth ： keevinzha
@File ：__init__.py.py
@IDE ：PyCharm
"""

import os
import importlib
import scipy.io as scio
import h5py
import mat73

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

from .base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def build_dataset(is_train, args):
    data_path = args.data_path if is_train else args.eval_data_path
    build_datafile_list(data_path, args.max_dataset_size)
    # todo 这里还没改完，这里应该返回数据集dataset
    dataset_class = find_dataset_using_name(args.dataset)
    dataset = dataset_class(args, is_train)

    return dataset


def build_datafile_list(root_dir, max_dataset_size=float("inf")):
    """
    Walk through all files in a directory, and return a list of all file paths.
    :param root_dir: path to data directory
    :param max_dataset_size: maximum number of datas, useful for debugging and running validation on subsets
    :return: a list of all file paths
    """
    datas = []
    assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in fnames:
            if '.mat' in fname:
                path = os.path.join(root, fname)
                datas.append(path)

    return datas[:min(max_dataset_size, len(datas))]


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def read_data(path, key:str):
    """
    Read data from path
    :param path: path of data
    :param key: key of data
    :return: data
    """
    try:
        data = scio.loadmat(path)
        data = data[key][:]
    except ValueError:
        f = h5py.File(path, 'r')
        data = f[key][:]
    return data
