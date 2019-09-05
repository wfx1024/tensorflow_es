# coding: utf-8

from __future__ import division, print_function

import cv2
import random


def get_color_table(class_num, seed=2):
    """
    多个类别生成不同颜色
    :param class_num: 类别数量
    :param seed:
    :return:
    """
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    """
    画出bbox方法
    :param img: 画框图片
    :param coord: [x_min, y_min, x_max, y_max] 格式化坐标
    :param label: 标签名
    :param color: 颜色index
    :param line_thickness: int. 框厚度.
    :return:
    """
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    line_type = cv2.LINE_AA if cv2.__version__ > '3' else cv2.LINE_AA
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=line_type)

