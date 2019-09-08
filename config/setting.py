# -*- coding:utf-8 -*-

from utils.misc_utils import parse_anchors, read_class_names
from utils.plot_utils import get_color_table

"""文件参数"""
class_name_path = './data/coco.names'  # 类别文件
classes = read_class_names(class_name_path)  # 类别
class_num = len(classes)  # 类别数量
anchors_path = './data/yolo_anchors.txt'  # anchor boxes文件
anchors = parse_anchors(anchors_path)  # anchor
weights_path = './data/darknet_weights/yolov3.ckpt'  # 权重文件

"""训练策略"""
letterbox_resize_used = True  # 是否使用letterbox训练
multi_scale_train = True  # 是否使用多尺度训练策略. 图片尺度从 [320, 320] 到 [640, 640]
use_label_smooth = False  # Whether to use class label smoothing strategy.
use_static_shape = True  # use_static_shape=True使用tensor.get_shape(),否则使用tf.shape(tensor) 固定大小会快一点
use_focal_loss = False  # 是否应用 focal loss on the conf loss.
use_mix_up = True  # 是否使用 mix up data augmentation strategy.
use_warm_up = True  # 是否使用 warm up strategy to prevent from gradient exploding.
reuse = False
warm_up_epoch = 3  # Warm up training epoches. Set to a larger value if gradient explodes.

"""NMS参数"""
nms_threshold = 0.45  # iou阈值
score_threshold = 0.3  # nms类别置信度, i.e. score = pred_confs * pred_probs. set lower for higher recall.
# score_thresh = 0.3
nms_topk = 150  # keep at most nms_topk outputs after nms

weight_decay = 5e-4
batch_norm_decay = 0.999

img_size = [416, 416]  # 传入net的图片大小(非输入图片大小)
max_boxes = 200  # 最大bbox数

color_table = get_color_table(class_num)  # 类别颜色
