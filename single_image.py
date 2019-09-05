# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3


def main(new_size, classes, anchors, restore_path, letterbox_resize):
    """
    入口函数
    :param new_size:
    :param classes:
    :param anchors:
    :param restore_path:
    :param letterbox_resize:
    :return:
    """
    num_class = len(classes)
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, restore_path)

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        if letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), line_thickness=3, color=color_table[labels_[i]])
        cv2.imshow('Detection result', img_ori)
        cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)


def parse_input():
    parser = argparse.ArgumentParser(description="YOLO_v3 单图测试文件")
    parser.add_argument("input_image", type=str, help="图片路径", default='./data/demo_data/kite.jpg')
    parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt", help="bbox信息txt文件")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416], help="检测后图片大小")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True, help="裁剪大小")
    parser.add_argument("--class_name_path", type=str, default="./data/coco.names", help="类别文件")
    parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt", help="权重路径")

    args = parser.parse_args()

    args.anchors = parse_anchors(args.anchor_path)
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)


# python single_image.py ./data/demo_data/kite.jpg
if __name__ == '__main__':
    anchors_path = './data/yolo_anchors.txt'  # k-means 得到的anchor box
    input_image = './data/demo_data/me.png'  # 输入图片
    classes = read_class_names('./data/coco.names')  # 类别文件
    restore_path = './data/darknet_weights/yolov3.ckpt'  # 权重文件
    num_class = len(classes)  # 类别数量
    letterbox_resize_used = True  # 裁剪大小
    new_size = [416, 416]

    color_table = get_color_table(num_class)
    anchors = parse_anchors(anchors_path)
    img_ori = cv2.imread(input_image)
    if letterbox_resize_used:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size[0], new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    main(new_size, classes, anchors, restore_path, letterbox_resize)
