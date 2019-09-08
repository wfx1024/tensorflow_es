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






class_name_path = './data/coco.names'  # 类别文件
classes = read_class_names(class_name_path)  # 类别
class_num = len(classes)  # 类别数量

anchor_path = './data/yolo_anchors.txt'  # anchor boxes文件
anchors = parse_anchors(anchor_path)  # anchor
weights_path = './data/darknet_weights/yolov3.ckpt'  # 权重文件

"""训练策略"""
multi_scale_train = True  # Whether to apply multi-scale training strategy. Image size varies from [320, 320] to [640, 640] by default.
use_label_smooth = False  # Whether to use class label smoothing strategy.
use_static_shape = True  # use_static_shape=True使用tensor.get_shape(),否则使用tf.shape(tensor) 固定大小会快一点
use_focal_loss = False  # Whether to apply focal loss on the conf loss.
use_mix_up = True  # Whether to use mix up data augmentation strategy.
use_warm_up = True  # whether to use warm up strategy to prevent from gradient exploding.
reuse = False
warm_up_epoch = 3  # Warm up training epoches. Set to a larger value if gradient explodes.


"""NMS参数"""
nms_threshold = 0.45  # iou阈值
# nms_thresh = 0.45
score_threshold = 0.01  # threshold of the probability of the classes in nms operation, i.e. score = pred_confs * pred_probs. set lower for higher recall.
# score_thresh = 0.3
nms_topk = 150  # keep at most nms_topk outputs after nms

weight_decay = 5e-4
batch_norm_decay = 0.999

img_size = [416, 416]  # 传入net的图片大小(非输入图片大小)
max_boxes = 200
