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
import matplotlib.pyplot as plt

from datasets import read_data
from util.recon_tools import ifft2c, sos
from util.utils import mat_to_yolo_input, yolobbox_to_mask

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


model = YOLO('../pretrained_models/best.pt')

# train
# results = model.train(data='../datasets/cardiac.yaml', epochs=100, device=[0,1,2,3])
# test
data = mat_to_yolo_input('/data/disk1/datasets/T1map_dataset/train/kspace_multicoil_128_singleslice/P001/2_T1map.mat')

ks = read_data('/data/disk1/datasets/T1map_dataset/train/kspace_multicoil_128_singleslice/P001/2_T1map.mat', 'kspace')
img = ifft2c(ks, (0, 1))
img = np.squeeze(sos(img, 2))
img = img.astype(np.float32)

results = model([data])
boxes = results[0].boxes.xyxy[0].cpu().numpy()
mask = yolobbox_to_mask(boxes, (128, 128))
plt.imshow(mask * img[:,:,0])
plt.show()