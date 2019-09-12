# coding: utf-8

from __future__ import division, print_function
from utils.misc_utils import parse_anchors, read_class_names
import math

"""
训练配置
"""

"""路径文件"""
input_image = './data/demo_data/dog.jpg'  # 输入图片
input_vedio = './data/demo_data/video_demo.mp4'  # 输入视频
train_file = './data/sample/train.txt'  # 训练文件路径 # ./data/my_data/wider_face_train_bbx_gt.txt
val_file = './data/sample/val.txt'  # 验证文件路径 # ./data/my_data/val.txt
restore_path = './data/darknet_weights/yolov3.ckpt'  # weights路径
save_dir = './checkpoint/'  # 保存weights路径
log_dir = './data/logs/'  # 保存tensorboard文件路径.
progress_log_path = './data/progress.log'  # 记录training progress文件路径
anchor_path = './data/yolo_anchors.txt'  # anchor路径文件
class_name_path = './data/coco.names'  # 类别文件路径

"""训练参数"""
batch_size = 6
img_size = [416, 416]  # Images will be resized to `img_size` and fed to the network, size format: [width, height]
letterbox_resize = True  # 是否使用letterbox
total_epoches = 100
train_evaluation_step = 100  # 训练集训练train_evaluation_step epochs后，使用验证集评估
val_evaluation_epoch = 2  # 训练集训练若干epochs后，使用验证集评估val_evaluation_epoch epochs
save_epoch = 1  # 多少epochs后保存模型
batch_norm_decay = 0.99  # decay in bn ops
weight_decay = 5e-4  # l2 weight decay
global_step = 0  # 重新train时使用

"""dataset参数"""
num_threads = 10  # tf.data读取时，线程数
prefetech_buffer = 5  # 每个butch取的图片数

"""学习率和优化器"""
optimizer_name = 'momentum'  # 优化器策略[sgd, momentum, adam, rmsprop]
save_optimizer = False  # 是否保存优化optimizer参数
learning_rate_init = 1e-4
lr_type = 'piecewise'  # Chosen from [fixed, exponential, cosine_decay, cosine_decay_restart, piecewise]
lr_decay_epoch = 5  # Epochs after which learning rate decays. Int or float. Used when chosen `exponential` and `cosine_decay_restart` lr_type.
lr_decay_factor = 0.96  # The learning rate decay factor. Used when chosen `exponential` lr_type.
lr_lower_bound = 1e-6  # The minimum learning rate.
# only used in piecewise lr type
pw_boundaries = [30, 50]  # epoch based boundaries
pw_values = [learning_rate_init, 3e-5, 1e-5]

"""fine tuning"""
# restore_include: None, restore_exclude: None  => 加载所有model的weights
# restore_include: None, restore_exclude: scope  =>加载所有model的weights, 除去scope
# restore_include: scope1, restore_exclude: scope2  => 加载(scope1 - scope2)的部分

# 策略 1: 只加载darknet53_body
# restore_include = ['yolov3/darknet53_body']
# restore_exclude = None
# 策略 2: 除了最后3个conv2d layers in 3 scale，加载所有layers
restore_include = None
restore_exclude = [
    'yolov3/yolov3_head/Conv_14',
    'yolov3/yolov3_head/Conv_6',
    'yolov3/yolov3_head/Conv_22'
]

# 选择想要fine tune的部分, None则fine-tune整个模型
update_part = ['yolov3/yolov3_head']

"""训练策略Tricks"""
multi_scale_train = True  # 是否使用multi-scale, 图片大小 [320, 320] 到 [640, 640]
use_label_smooth = True  # 是否使用label smoothing(gt分布混合)
use_focal_loss = True  # focal loss on the conf loss.
use_mix_up = True  # 是否用mix up(数据增强, 抗扰动)
use_warm_up = True  # 是否使用warm up(预热训练)，先使用小学习率，一定epoch后再用大的学习率
warm_up_epoch = 3  # warm up训练多少epoch

"""验证参数"""
# nms
nms_threshold = 0.45  # nms 时的iou
score_threshold = 0.01  # nms类别置信度, score = pred_confs * pred_probs. set lower for higher recall.
nms_topk = 150  # keep at most nms_topk outputs after nms
# mAP eval
eval_threshold = 0.5  # the iou threshold applied in mAP evaluation
use_voc_07_metric = False  # whether to use voc 2007 evaluation metric, i.e. the 11-point metric

"""解析参数"""
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
class_num = len(classes)
train_img_cnt = len(open(train_file, 'r').readlines())  # 训练集样本数
val_img_cnt = len(open(val_file, 'r').readlines())  # 验证集样本数
train_batch_num = int(math.ceil(float(train_img_cnt) / batch_size))  # batch数

lr_decay_freq = int(train_batch_num * lr_decay_epoch)
pw_boundaries = [float(i) * train_batch_num + global_step for i in pw_boundaries]