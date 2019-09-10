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


def get_strategy(is_training):
    """
    # predict: 默认
    # use_label_smooth=False,use_focal_loss=False,batch_norm_decay=0.999,weight_decay=5e-4,use_static_shape=True
    # train:
    # use_label_smooth=True,use_focal_loss=True,batch_norm_decay = 0.99,weight_decay=5e-4,use_static_shape=False
    :param is_training:
    :return:
    """
    use_label_smooth = False if not is_training else True  # 是否使用 class label smoothing strategy.
    use_focal_loss = False if not is_training else True  # 是否应用 focal loss on the conf loss.
    use_static_shape = True if not is_training else False  # True时 使用tensor.get_shape(),否则使用tf.shape(tensor) 固定大小会快一点
    batch_norm_decay = 0.999 if not is_training else 0.99
    return use_label_smooth, use_focal_loss, use_static_shape, batch_norm_decay


def get_score_threshold(is_training):
    score_threshold = 0.3 if not is_training else 0.01
    return score_threshold


reuse = False
letterbox_resize_used = True  # 是否使用letterbox训练

weight_decay = 5e-4



"""NMS参数"""
nms_threshold = 0.45  # iou阈值
  # nms类别置信度, i.e. score = pred_confs * pred_probs. set lower for higher recall.
# train_score_threshold = 0.01



img_size = [416, 416]  # 传入net的图片大小(非输入图片大小)
max_boxes = 200  # 最大bbox数

color_table = get_color_table(class_num)  # 类别颜色
