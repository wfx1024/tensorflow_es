# -*- coding:utf-8 -*-
from utils.misc_utils import parse_anchors, read_class_names
from utils.plot_utils import get_color_table

"""
检测配置
"""

detect_object = 'img'  # 默认检测对象
input_image = './data/demo_data/person.jpg'  # 默认图片路径
input_video = './data/demo_data/video_demo.mp4'  # 默认视频路径
output_image = './data/demo_data/result/result.jpg'  # 保存图片路径
output_video = './data/demo_data/result/result.mp4'  # 保存视频路径
anchor_path = './data/yolo_anchors.txt'  # anchor 文件路径
anchors = parse_anchors(anchor_path)  # anchor内容
weight_path = './data/darknet_weights/yolov3.ckpt'  # weights路径

class_name_path = './data/coco.names'  # 类别文件路径
classes = read_class_names(class_name_path)  # 类别文件list
num_class = len(classes)  # 类别数量

new_size = [416, 416]  # 图片改变后的大小
use_letterbox_resize = True  # 是否使用letterbox
color_table = get_color_table(num_class)  # 根据类别数生成颜色列表
