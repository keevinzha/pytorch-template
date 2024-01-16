# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/20 13:22
@Auth ： keevinzha
@File ：base_model.py
@IDE ：PyCharm
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def get_loss_fn(self):
        """
        This is the interface for obtaining the loss function in the main function.
        :return: loss function
        """
        return torch.nn.MSELoss()

    def get_metrics(self):
        """
        This is the interface for obtaining the metrics in the main function.
        :return: metrics
        """
        return torch.nn.MSELoss()
