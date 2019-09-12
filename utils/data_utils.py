# coding: utf-8

from __future__ import division, print_function
import sys
from utils.data_aug import *
import random
import tensorflow as tf
from setting import train_args as args

"""
数据解析相关
"""

PY_VERSION = sys.version_info[0]
iter_cnt = 0


def parse_line(line):
    """
    解析每行(COCO数据集格式)
    :param line: 格式: line_idx File_name x1 y1 w1 h1 label x2 y2 w2 h2 label x3 y3 w3 h3 label ...
    :return:
        line_idx:行数,
        pic_path: 图片路径
        boxes: [N, 4], N 是GT数量, 4为[x_min, y_min, x_max, y_max]
        labels: [N]. 类别id.
        img_width, img_height: 图片大小
    """
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(s) > 8, '一个图片至少有一个bbox, 检查标注'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(s) % 5 == 0, 'bbox至少有5个值, 4-坐标和1-类别, 检查标注'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max \
            = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels, img_width, img_height


def process_box(boxes, labels, img_size, class_num, anchors):
    """
    生成 y_true label, 也就是gt在三种不同维度维度上的feature_maps
    :param boxes: float32, [N, 5] x_min, y_min, x_max, y_mix, mixup_weight(混合程度).
    :param labels: int64 [N] shape
    :param img_size:
    :param class_num: int64 num.
    :param anchors: [9, 2] float32
    :return:
    """
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # 改变boxes格式: [N, 2], 得到中心点相对值坐标(x_center, y_center), 缩放无影响
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # 416*416举例, 416/31=13: [13, 13, 3, 5+num_class+1]=[13, 13, 3, 86]
    # 5是坐标和类别标签, 1是mix_up weight(混合程度)
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + class_num), np.float32)

    # mix_up weight默认为1.
    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 2]-->[N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # 广播: [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins
    # [N, 9] IoU
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10)
    # [N]
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        print("[", i, idx, feature_map_group, "]  ", y, x, k, c)
        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode, use_letterbox_resize):
    """
    解析每行数据到y_true
    :param line: a line from the training/test txt file
    :param class_num: totol class nums.
    :param img_size:  the size of image to be resized to. [width, height] format.
    :param anchors: anchors.
    :param mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    :param use_letterbox_resize:  whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
    :return:
    """
    print("line:{}\n\n".format(line))
    # 如果一条，则直接解析
    if not isinstance(line, list):
        img_idx, pic_path, boxes, labels, _, _ = parse_line(line)
        img = cv2.imread(pic_path)
        # expand the 2nd dimension, mix up weight default to 1.
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
    # 如果两条，则mix up混合
    else:
        _, pic_path1, boxes1, labels1, _, _ = parse_line(line[0])
        img1 = cv2.imread(pic_path1)
        img_idx, pic_path2, boxes2, labels2, _, _ = parse_line(line[1])
        img2 = cv2.imread(pic_path2)

        img, boxes = mix_up(img1, img2, boxes1, boxes2)
        labels = np.concatenate((labels1, labels2))

    # 如果train, 对解析到的img应用各种tricks(bbox随之调整)
    if mode == 'train':
        # 扭曲图片
        img = random_color_distort(img)

        # 50%几率，应用随机放大
        if np.random.uniform(0, 1) > 0.5:
            img, boxes = random_expand(img, boxes, 4)

        # 随机裁剪
        h, w, _ = img.shape
        boxes, crop = random_crop_with_constraints(boxes, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0+h, x0: x0+w]

        # 调整图片大小
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, boxes = resize_with_bbox(
            img, boxes, img_size[0], img_size[1], interp=interp, letterbox=use_letterbox_resize
        )

        # 随机滑动
        h, w, _ = img.shape
        img, boxes = random_flip(img, boxes, px=0.5)
    # 否则直接将图片调整到输入大小
    else:
        img, boxes = resize_with_bbox(
            img, boxes, img_size[0], img_size[1], interp=1, letterbox=use_letterbox_resize
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 255.   # v3要求值归一化[0, 255]-->[0, 1]
    # 得到在三种维度上的gt
    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)
    return img_idx, img, y_true_13, y_true_26, y_true_52


def get_batch_data(batch_line, class_num, img_size, anchors, mode, multi_scale=False, mix_up=False, letterbox_resize=True, interval=10):
    """
    获得批数据(imgs和labels)
    :param batch_line: batch数量的line
    :param class_num: 类别数
    :param img_size: 416*416
    :param anchors: anchors. shape=[9, 2].
    :param mode: train或val.如果是train, 应用data augmentation
    :param multi_scale: 是否multi_scale training, 图片大小 32*[10->20]也就是[320, 320]->[640, 640]，mode=train有效
    :param mix_up:
    :param letterbox_resize: 是否letterbox resize, i.e., keep the original aspect ratio in the resized image.
    :param interval: change the scale of image every interval batches. Note that it's indeterministic because of the multi threading.
    :return:
    """
    global iter_cnt
    # multi_scale 训练
    if multi_scale and mode == 'train':
        random.seed(iter_cnt // interval)
        random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
        img_size = random.sample(random_img_size, 1)[0]
    iter_cnt += 1

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

    # train且mix up
    if mix_up and mode == 'train':
        mix_lines = []
        batch_line = batch_line.tolist()
        for idx, line in enumerate(batch_line):
            if np.random.uniform(0, 1) < 0.5:
                # 在当前batch中获取另一个line
                mix_lines.append([line, random.sample(batch_line[:idx] + batch_line[idx+1:], 1)[0]])
            else:
                mix_lines.append(line)
        batch_line = mix_lines
        print("batch:{}\n".format(batch_line))

    for line in batch_line:
        img_idx, img, y_true_13, y_true_26, y_true_52 = parse_data(line, class_num, img_size, anchors, mode, letterbox_resize)

        img_idx_batch.append(img_idx)
        img_batch.append(img)
        y_true_13_batch.append(y_true_13)
        y_true_26_batch.append(y_true_26)
        y_true_52_batch.append(y_true_52)

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = np.asarray(img_idx_batch, np.int64), np.asarray(img_batch), np.asarray(y_true_13_batch), np.asarray(y_true_26_batch), np.asarray(y_true_52_batch)

    return img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch


def build_train_dataset():
    """
    构建验证数据
    :return:
    """
    train_dataset = tf.data.TextLineDataset(args.train_file)
    train_dataset = train_dataset.shuffle(args.train_img_cnt)  # 先随机重排
    train_dataset = train_dataset.batch(args.batch_size)  # 分批
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(
            get_batch_data,
            inp=[x, args.class_num, args.img_size, args.anchors, 'train',
                 args.multi_scale_train, args.use_mix_up, args.letterbox_resize],
            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    train_dataset = train_dataset.prefetch(args.prefetech_buffer)  # 每次取1
    return train_dataset


def build_val_dataset():
    """
    构建验证数据集
    :return:
    """
    val_dataset = tf.data.TextLineDataset(args.val_file)
    val_dataset = val_dataset.batch(1)  # 一批一个
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(
            get_batch_data,
            inp=[x, args.class_num, args.img_size, args.anchors,
                 'val', False, False, args.letterbox_resize],
            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    val_dataset = val_dataset.prefetch(args.prefetech_buffer)
    return val_dataset


def create_iterator():
    """
    创建迭代器
    :return:
    """
    print('\n\033[32m----------- Begin building dataset  -----------\n')
    train_dataset = build_train_dataset()  # 训练集
    val_dataset = build_val_dataset()  # 验证集
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    # 获得一条数据
    image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    # 如果丢失了shape,则手动设置
    image_ids.set_shape([None])
    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])
    print('\n\033[32m----------- Finish building dataset  -----------\n')
    return train_init_op, val_init_op, image_ids, image, y_true
