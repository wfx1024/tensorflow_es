# -*- coding:utf-8 -*-
import numpy as np
import cv2

"""
图片预处理，处理tricks
"""


def letterbox_resize(img, new_width, new_height, interp=0):
    """
    以改变较小的边为基准，等比改变大小，另一边填充灰边
    :param img:
    :param new_width:
    :param new_height:
    :param interp:
    :return:
    """
    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)  # 改变比例，选改变小的
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)
    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)
    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    return image_padded, resize_ratio, dw, dh
