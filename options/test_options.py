# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/17 13:20
@Auth ： keevinzha
@File ：test_options.py
@IDE ：PyCharm
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    This class includes test options.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.isTrain = False
        return parser