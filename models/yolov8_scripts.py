# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/3 16:41
@Auth ： keevinzha
@File ：yolov8_scripts.py
@IDE ：PyCharm
"""

from ultralytics import YOLO
import numpy as np
import os
import cv2

from datasets import read_data
from util.recon_tools import ifft2c, sos

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


model = YOLO('../pretrained_models/yolov8n.pt')


# results = model.train(data='../datasets/cardiac.yaml', epochs=100)
results = model.predict()