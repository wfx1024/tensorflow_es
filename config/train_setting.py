# -*- coding:utf-8 -*-

from utils.misc_utils import parse_anchors, read_class_names
import math


"""
训练过程参数
"""

"""初始化参数"""
letterbox_resize = True  # 是否缩放
total_epoches = 100  # epoch数
batch_size = 6  # batch 大小
img_size = [416, 416]  # 传入net的图片大小

"""样本相关"""
train_file = './data/sample/train.txt'  # 训练样本列表txt文件
val_file = './data/sample/val.txt'  # 验证样本列表txt文件
class_name_path = './data/coco.names'  # 类别文件
classes = read_class_names(class_name_path)  # 类别
class_num = len(classes)  # 类别数量
train_img_cnt = len(open(train_file, 'r').readlines())  # 测试集行数
val_img_cnt = len(open(val_file, 'r').readlines())  # 验证集行数
train_batch_num = int(math.ceil(float(train_img_cnt) / batch_size))  # batch数

"""存储数据路径"""
anchor_path = './data/yolo_anchors.txt'  # anchor boxes文件
anchors = parse_anchors(anchor_path)  # anchor
weights_path = './data/darknet_weights/yolov3.ckpt'  # 权重文件
save_dir = './checkpoint/'  # 权重保存
log_dir = './data/logs/'  # log文件存储
progress_log_path = './data/progress.log'  # 训练日志


train_evaluation_step = 100  # Evaluate on the training batch after some steps.
val_evaluation_epoch = 2  # Evaluate on the whole validation dataset after some epochs. Set to None to evaluate every epoch.
save_epoch = 10  # Save the model after some epochs.
batch_norm_decay = 0.99  # decay in bn ops
weight_decay = 5e-4  # l2 weight decay
global_step = 0  # used when resuming training

# tf.data参数
num_threads = 10  # tf.data pipeline线程数
prefetech_buffer = 5  # Prefetech_buffer used in tf.data pipeline.

"""训练参数"""
optimizer_name = 'momentum'  # Chosen from [sgd, momentum, adam, rmsprop]
save_optimizer = True  # Whether to save the optimizer parameters into the checkpoint file.
learning_rate_init = 1e-4
lr_type = 'piecewise'  # Chosen from [fixed, exponential, cosine_decay, cosine_decay_restart, piecewise]
lr_decay_epoch = 5  # Epochs after which learning rate decays. Int or float. Used when chosen `exponential` and `cosine_decay_restart` lr_type.
lr_decay_factor = 0.96  # The learning rate decay factor. Used when chosen `exponential` lr_type.
lr_lower_bound = 1e-6  # The minimum learning rate.
# only used in piecewise lr type
pw_boundaries = [30, 50]  # epoch based boundaries
pw_values = [learning_rate_init, 3e-5, 1e-5]

"""训练策略"""
multi_scale_train = True  # Whether to apply multi-scale training strategy. Image size varies from [320, 320] to [640, 640] by default.
use_label_smooth = True  # Whether to use class label smoothing strategy.
use_focal_loss = True  # Whether to apply focal loss on the conf loss.
use_mix_up = True  # Whether to use mix up data augmentation strategy.
use_warm_up = True  # whether to use warm up strategy to prevent from gradient exploding.
warm_up_epoch = 3  # Warm up training epoches. Set to a larger value if gradient explodes.

"""NMS参数"""
nms_threshold = 0.45  # iou阈值
score_threshold = 0.01  # threshold of the probability of the classes in nms operation, i.e. score = pred_confs * pred_probs. set lower for higher recall.
nms_topk = 150  # keep at most nms_topk outputs after nms

# mAP eval
eval_threshold = 0.5  # the iou threshold applied in mAP evaluation
use_voc_07_metric = False  # whether to use voc 2007 evaluation metric, i.e. the 11-point metric


lr_decay_freq = int(train_batch_num * lr_decay_epoch)
pw_boundaries = [float(i) * train_batch_num + global_step for i in pw_boundaries]

### Load and finetune
# Choose the parts you want to restore the weights. List form.
# restore_include: None, restore_exclude: None  => restore the whole model
# restore_include: None, restore_exclude: scope  => restore the whole model except `scope`
# restore_include: scope1, restore_exclude: scope2  => if scope1 contains scope2, restore scope1 and not restore scope2 (scope1 - scope2)
# choise 1: only restore the darknet body
# restore_include = ['yolov3/darknet53_body']
# restore_exclude = None
# choise 2: restore all layers except the last 3 conv2d layers in 3 scale
restore_include = None
restore_exclude = ['yolov3/yolov3_head/Conv_14', 'yolov3/yolov3_head/Conv_6', 'yolov3/yolov3_head/Conv_22']
# Choose the parts you want to finetune. List form.
# Set to None to train the whole model.
update_part = ['yolov3/yolov3_head']