# coding: utf-8

from __future__ import division, print_function

import config.predict_setting as config
import tensorflow as tf
import numpy as np
import argparse
import cv2
from utils.nms_utils import gpu_nms
from utils.plot_utils import plot_one_box
from utils.data_aug import letterbox_resize
from model import YoloV3


def predict(weights_path, img_origin):
    """
    图片预测
    :param weights_path:
    :param img_origin:
    :return:
    """
    resize_ratio = 1
    dw = dh = 0
    width_origin = height_origin = 0
    if config.letterbox_resize_used:
        img, resize_ratio, dw, dh = letterbox_resize(img_origin, config.img_size[0], config.img_size[1])
    else:
        height_origin, width_origin = img_origin.shape[:2]
        img = cv2.resize(img_origin, tuple(config.img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, config.img_size[1], config.img_size[0], 3], name='input_data')
        yolo_model = YoloV3(config.num_class, config.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(
            pred_boxes, pred_scores, config.num_class, max_boxes=config.max_boxes,
            score_thresh=config.score_thresh, nms_thresh=config.nms_thresh
        )

        saver = tf.train.Saver()
        saver.restore(sess, weights_path)

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        if config.letterbox_resize_used:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_origin/float(config.img_size[0]))
            boxes_[:, [1, 3]] *= (height_origin/float(config.img_size[1]))

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
            plot_one_box(img_origin, [x0, y0, x1, y1], label=config.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), line_thickness=3, color=config.color_table[labels_[i]])
        cv2.imshow('Detection result', img_origin)
        cv2.imwrite('detection_result.jpg', img_origin)
    cv2.waitKey(0)


def main():
    # 如果有输入则使用输入，否则使用config/predict_setting.py中的默认参数
    parser = argparse.ArgumentParser(description="YOLO_v3 图片测试文件")
    parser.add_argument("--input_image", type=str, help="图片路径", default=config.input_image)
    parser.add_argument("--weights_path", type=str, default=config.weights_path, help="权重路径")

    args = parser.parse_args()
    img_origin = cv2.imread(args.input_image)  # 原始图片
    predict(args.weights_path, img_origin)


if __name__ == '__main__':
    main()
