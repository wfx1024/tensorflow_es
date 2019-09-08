# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import config.setting as setting
import config.predict_setting as predict_setting
from utils.plot_utils import plot_one_box
from utils.data_aug import letterbox_resize
from net.yolov3 import YoloV3


# def plot_bbox(img, bboxes, scores, labels):
#     """
#
#     :return:
#     """
#     print("bbox coords:", bboxes, '*' * 30, "scores:", scores, '*' * 30, "labels:", labels)
#
#     for i in range(len(bboxes)):
#         x0, y0, x1, y1 = bboxes[i]
#         plot_one_box(
#             img, [x0, y0, x1, y1],
#             label=predict_setting.classes[labels[i]] + ', {:.2f}%'.format(scores[i] * 100),
#             line_thickness=3, color=predict_setting.color_table[labels[i]]
#         )
#     cv2.imshow('Detection result', img)
#     cv2.imwrite(predict_setting.save_img_path, img)
#     cv2.waitKey(0)


def detect(yolov3, img_origin):
    """
    检测方法
    :return:
    """
    resize_ratio = 1
    dw = dh = 0
    width_origin = height_origin = 0
    if predict_setting.letterbox_resize_used:
        img, resize_ratio, dw, dh = letterbox_resize(img_origin, setting.img_size[0], setting.img_size[1])
    else:
        height_origin, width_origin = img_origin.shape[:2]
        img = cv2.resize(img_origin, tuple(setting.img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    bboxes, scores, labels = yolov3.predict(img)

    # 对原图调节坐标
    if predict_setting.letterbox_resize_used:
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / resize_ratio
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / resize_ratio
    else:
        bboxes[:, [0, 2]] *= (width_origin / float(setting.img_size[0]))
        bboxes[:, [1, 3]] *= (height_origin / float(setting.img_size[1]))

    print("bbox coords:", bboxes, '*' * 30, "scores:", scores, '*' * 30, "labels:", labels)

    for i in range(len(bboxes)):
        x0, y0, x1, y1 = bboxes[i]
        plot_one_box(
            img_origin, [x0, y0, x1, y1],
            label=predict_setting.classes[labels[i]] + ', {:.2f}%'.format(scores[i] * 100),
            color=predict_setting.color_table[labels[i]]  # line_thickness=2,
        )
    return img_origin


def img_detect2(img_origin):
    with tf.variable_scope('yolov3'):
        yolov3 = YoloV3()
    img_origin = detect(yolov3, img_origin)
    cv2.imwrite(predict_setting.save_img_path, img_origin)
    cv2.imshow('img', img_origin)
    cv2.waitKey(0)
    return img_origin


def vedio_detect2(input_video):
    vid = cv2.VideoCapture(input_video)
    video_writer = cv2.VideoWriter(
        'video_result.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        int(vid.get(5)), (int(vid.get(3)), int(vid.get(4)))  # fps, width, height
    )

    start_time = time.time()
    with tf.variable_scope('yolov3'):
        yolov3 = YoloV3()
    yolov3_time = time.time()
    print("yolo v3 build time:%ss" % str(yolov3_time - start_time))
    for _ in range(int(vid.get(7))):
        ret, frame = vid.read()
        start_time = time.time()
        img = detect(yolov3, frame)
        end_time = time.time()
        cv2.putText(
            img, '%s ms' % str(int((end_time - start_time) * 1000)),
            (20, 20), 0, fontScale=0.5, color=(0, 255, 0)
        )
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        video_writer.write(img)
    vid.release()
    video_writer.release()


def main():
    # 如果有输入则使用输入，否则使用config/predict_setting.py中的默认参数
    parser = argparse.ArgumentParser(description="YOLO_v3 图片测试文件")
    parser.add_argument("--file_type", type=str, default=predict_setting.file_type, help="视频还是图片")
    parser.add_argument("--input_image", type=str, help="图片路径", default=predict_setting.input_image)
    parser.add_argument("--input_vedio", type=str, help="视频路径", default=predict_setting.input_vedio)
    # parser.add_argument("--weights_path", type=str, default=config.weights_path, help="权重路径")
    args = parser.parse_args()

    if args.file_type == 'img':
        img_origin = cv2.imread(args.input_image)  # 原始图片
        img_detect2(img_origin)
    elif args.file_type == 'vedio':
        vedio_detect2(args.input_vedio)


if __name__ == '__main__':
    main()
