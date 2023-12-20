# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/7 09:21
@Auth ： keevinzha
@File ：organize_yolo_dataset.py
@IDE ：PyCharm
"""
import os

from options.process_data_options import ProcessDataOptions


def organize_yolo_dataset(yolo_image_root, yolo_label_root, jpg_root, bbox_root):
    """
    :param yolo_image_root: path to yolo image
    :param yolo_label_root: path to yolo label
    :param jpg_root: path to jpg
    :return:
    """
    volunteer_list = []
    prefix = 'train'
    for base_dir, dirs, files in os.walk(jpg_root):
        if not dirs:
            for file in files:
                if '.jpg' in file:
                    base_name = os.path.basename(base_dir)
                    jpg_path = os.path.join(base_dir, file)
                    bbox_path = os.path.join(bbox_root, base_name, file.replace('.jpg', '.txt'))
                    # seperate train and val
                    if(base_name not in volunteer_list):
                        volunteer_list.append(base_name)
                    if len(volunteer_list) > 100:
                        prefix = 'val'
                    elif len(volunteer_list) > 110:
                        prefix = 'test'


                    yolo_image_path = os.path.join(yolo_image_root, prefix, base_name + file)

                    yolo_label_path = os.path.join(yolo_label_root, prefix, base_name + file.replace('.jpg', '.txt'))  # 生成对应图片名字的bbox文件
                    # copy jpg
                    os.system('cp {} {}'.format(jpg_path, yolo_image_path))
                    # copy bbox
                    os.system('cp {} {}'.format(bbox_path, yolo_label_path))

if __name__ == '__main__':
    opt = ProcessDataOptions().parse()
    organize_yolo_dataset(opt.yolo_image_root, opt.yolo_label_root, opt.jpg_root, opt.bbox_root)
