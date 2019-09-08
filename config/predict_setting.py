# -*- coding:utf-8 -*-

from utils.misc_utils import read_class_names, parse_anchors
from utils.plot_utils import get_color_table

"""
预测过程参数
"""
file_type = 'vedio'  # img or vedio
anchors_path = './data/yolo_anchors.txt'  # k-means 得到的anchor box
anchors = parse_anchors(anchors_path)  # anchor
input_image = './data/demo_data/me.png'  # 输入图片
input_vedio = './data/demo_data/1.mp4'  # 输入视频
class_name_path = './data/coco.names'  # 类别文件
classes = read_class_names(class_name_path)  # 类别
num_class = len(classes)  # 类别数量
weights_path = './data/darknet_weights/yolov3.ckpt'  # 权重文件
letterbox_resize_used = True  # 裁剪大小

is_true = lambda x: (str(x).lower() == 'true')
color_table = get_color_table(num_class)
max_boxes = 200
score_thresh = 0.3
nms_thresh = 0.45

save_img_path = 'detection_result.jpg'