# -*- coding:utf-8 -*-
import tensorflow as tf
import config.setting as setting
import config.train_setting as train_setting
from utils.img_tricks import letterbox_resize
from utils.data_utils import get_batch_data
import sys
import random
import numpy as np
import cv2
"""
dataset处理tricks
"""

PY_VERSION = sys.version_info[0]
iter_cnt = 0


def random_flip(img, bbox, px=0, py=0):
    """
    Randomly flip the image and correct the bbox.
    :param img:
    :param bbox:
    :param px:the probability of horizontal flip
    :param py:the probability of vertical flip
    :return:
    """
    height, width = img.shape[:2]
    if np.random.uniform(0, 1) < px:
        img = cv2.flip(img, 1)
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    if np.random.uniform(0, 1) < py:
        img = cv2.flip(img, 0)
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    return img, bbox


def bbox_crop(bbox, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.
    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox


def mix_up(img1, img2, bbox1, bbox2):
    """
    融合
    :param img1:
    :param img2:
    :param bbox1:
    :param bbox2:
    :return:
        mix_img: HWC format mix up image
        mix_bbox: [N, 5] shape mix up bbox, i.e. `x_min, y_min, x_max, y_mix, mixup_weight`.
    """
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    mix_img = np.zeros(shape=(height, width, 3), dtype='float32')

    # rand_num = np.random.random()
    rand_num = np.random.beta(1.5, 1.5)
    rand_num = max(0, min(1, rand_num))
    mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * rand_num
    mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - rand_num)

    mix_img = mix_img.astype('uint8')

    # the last element of the 2nd dimention is the mix up weight
    bbox1 = np.concatenate((bbox1, np.full(shape=(bbox1.shape[0], 1), fill_value=rand_num)), axis=-1)
    bbox2 = np.concatenate((bbox2, np.full(shape=(bbox2.shape[0], 1), fill_value=1. - rand_num)), axis=-1)
    mix_bbox = np.concatenate((bbox1, bbox2), axis=0)

    return mix_img, mix_bbox


def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def resize_with_bbox(img, bbox, new_width, new_height, interp=0, letterbox_resize_used=False):
    """
     Resize the image and correct the bbox accordingly.
    :param img:
    :param bbox:
    :param new_width:
    :param new_height:
    :param interp:
    :param letterbox_resize_used:
    :return:
    """

    if letterbox_resize_used:
        image_padded, resize_ratio, dw, dh = letterbox_resize(img, new_width, new_height, interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * resize_ratio + dw
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * resize_ratio + dh

        return image_padded, bbox
    else:
        ori_height, ori_width = img.shape[:2]

        img = cv2.resize(img, (new_width, new_height), interpolation=interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * new_width
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * new_height

        return img, bbox


def random_expand(img, bbox, max_ratio=4, fill=0, keep_ratio=True):
    """
    Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.
    :param img:
    :param bbox:
    :param max_ratio: Maximum ratio of the output image on both direction(vertical and horizontal)
    :param fill:The value(s) for padded borders.
    :param keep_ratio: bool, If `True`, will keep output image the same aspect ratio as input.
    :return:
    """
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=img.dtype)

    dst[off_y:off_y + h, off_x:off_x + w, :] = img

    # correct bbox
    bbox[:, :2] += (off_x, off_y)
    bbox[:, 2:4] += (off_x, off_y)

    return dst, bbox

def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    """Crop an image randomly with bounding box constraints.
    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).
    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right-left, bottom-top)

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = bbox_crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)



def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    """
     randomly distort image color. Adjust brightness, hue, saturation, value.
    :param img:  a BGR uint8 format OpenCV image. HWC format.
    :param brightness_delta:
    :param hue_vari:
    :param sat_vari:
    :param val_vari:
    :return:
    """
    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def parse_line(line):
    """
    解析每行(COCO)
    line format:
    :param line: 格式 line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
    :return: line_idx, pic_path:
        boxes: shape [N, 4], N 是GT数量, 4为[x_min, y_min, x_max, y_max]
        labels: shape [N]. 类别id.
        img_width: int.
        img_height: int
    """
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(s) > 8, 'Annotation error, one image one bbox at least.'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(s) % 5 == 0, 'Annotation error, bbox param must be 5.'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels, img_width, img_height


def process_box(boxes, labels, img_size, class_num, anchors):
    """
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    :param boxes: [N, 5] shape, float32 dtype. `x_min, y_min, x_max, y_mix, mixup_weight`.
    :param labels: [N] shape, int64 dtype.
    :param img_size:
    :param class_num: int64 num.
    :param anchors: [9, 4] shape, float32 dtype.
    :return:
    """
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight.
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + class_num), np.float32)

    # mix up weight default to 1.
    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                         1] + 1e-10)
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
        # print(feature_map_group, '|', y,x,k,c)

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode, letterbox_resize_used):
    """
    解析每行数据
    :param line: a line from the training/test txt file
    :param class_num: totol class nums.
    :param img_size:  the size of image to be resized to. [width, height] format.
    :param anchors: anchors.
    :param mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    :param letterbox_resize_used:  whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
    :return:
    """
    if not isinstance(line, list):
        img_idx, pic_path, boxes, labels, _, _ = parse_line(line)
        img = cv2.imread(pic_path)
        # expand the 2nd dimension, mix up weight default to 1.
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
    else:
        # 两图以上可以mix up增强
        _, pic_path1, boxes1, labels1, _, _ = parse_line(line[0])
        img1 = cv2.imread(pic_path1)
        img_idx, pic_path2, boxes2, labels2, _, _ = parse_line(line[1])
        img2 = cv2.imread(pic_path2)

        img, boxes = mix_up(img1, img2, boxes1, boxes2)
        labels = np.concatenate((labels1, labels2))

    if mode == 'train':  # 如果训练，增加一系列trick
        # 随机颜色扰动(color jittering), 失真有时会导致坏结果
        img = random_color_distort(img)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, boxes = random_expand(img, boxes, 4)

        # 随机裁剪
        h, w, _ = img.shape
        boxes, crop = random_crop_with_constraints(boxes, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0+h, x0: x0+w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=interp, letterbox_resize_used=letterbox_resize_used)

        # random horizontal flip
        h, w, _ = img.shape
        img, boxes = random_flip(img, boxes, px=0.5)
    else:
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=1, letterbox_resize_used=letterbox_resize_used)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 255.  # v3 必须归一化输入

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)

    return img_idx, img, y_true_13, y_true_26, y_true_52


def get_batch_data(
        batch_line, class_num, img_size, anchors, mode, multi_scale=False,
        mix_up=False, letterbox_resize_used=True, interval=10):
    """
    获得批数据(imgs和labels)
    :param batch_line: batch数量的line
    :param class_num: 类别数
    :param img_size: 416*416
    :param anchors: anchors. shape=[9, 2].
    :param mode: train或val.如果是train, 应用data augmentation
    :param multi_scale: 是否multi_scale training, 图片大小 32*[10->20]也就是[320, 320]->[640, 640]，mode=train有效
    :param mix_up:
    :param letterbox_resize_used: 是否letterbox resize, i.e., keep the original aspect ratio in the resized image.
    :param interval: change the scale of image every interval batches. Note that it's indeterministic because of the multi threading.
    :return:
    """
    global iter_cnt
    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

    # 多尺度训练
    if multi_scale and mode.decode() == 'train':  # multi-Scale
        random.seed(iter_cnt // interval)
        random_img_size = [[x * 32, x * 32] for x in range(10, 20)]  # 32的10-20倍数
        img_size = random.sample(random_img_size, 1)[0]  # 选一个
    iter_cnt += 1

    # 融合策略
    if mix_up and mode.decode() == 'train':
        mix_lines = []
        batch_line = batch_line.tolist()
        for idx, line in enumerate(batch_line):
            # 随机2选1
            if np.random.uniform(0, 1) < 0.5:
                mix_lines.append([line, random.sample(batch_line[:idx] + batch_line[idx+1:], 1)[0]])
            else:
                mix_lines.append(line)
        batch_line = mix_lines

    # 解析每行
    for line in batch_line:
        img_idx, img, y_true_13, y_true_26, y_true_52 = parse_data(
            line, class_num, img_size, anchors, mode, letterbox_resize_used
        )
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
    train_dataset = tf.data.TextLineDataset(train_setting.train_file)
    train_dataset = train_dataset.shuffle(train_setting.train_img_cnt)  # 先随机重排
    train_dataset = train_dataset.batch(train_setting.batch_size)  # 分批
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(
            get_batch_data,
            inp=[x, setting.class_num, setting.img_size, setting.anchors, 'train',
                 train_setting.multi_scale_train, train_setting.use_mix_up, setting.letterbox_resize_used],
            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=train_setting.num_threads
    )
    train_dataset = train_dataset.prefetch(train_setting.prefetech_buffer)  # 每次取1
    return train_dataset


def build_val_dataset():
    """
    构建验证数据集
    :return:
    """
    val_dataset = tf.data.TextLineDataset(train_setting.val_file)
    val_dataset = val_dataset.batch(1)  # 一批一个
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(
            get_batch_data,
            inp=[x, setting.class_num, setting.img_size, setting.anchors, 'val',
                 False, False, setting.letterbox_resize_used],
            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=train_setting.num_threads
    )
    val_dataset = val_dataset.prefetch(train_setting.prefetech_buffer)
    return val_dataset


def create_iterator():
    """
    创建迭代器
    :return:
    """
    print('\n----------- Begin building dataset  -----------\n')
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
    print('\n----------- Finish building dataset  -----------\n')
    return train_init_op, val_init_op, image_ids, image, y_true
