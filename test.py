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


def img_detect(img_origin):
    """
    图片预测
    :param img_origin:
    :return:
    """
    with tf.variable_scope('yolov3'):
        yolov3 = YoloV3()

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
        bboxes[:, [0, 2]] *= (width_origin/float(setting.img_size[0]))
        bboxes[:, [1, 3]] *= (height_origin/float(setting.img_size[1]))

    print("bbox coords:", bboxes, '*' * 30, "scores:", scores, '*' * 30, "labels:", labels)

    for i in range(len(bboxes)):
        x0, y0, x1, y1 = bboxes[i]
        plot_one_box(
            img_origin, [x0, y0, x1, y1], label=predict_setting.classes[labels[i]] + ', {:.2f}%'.format(scores[i] * 100),
            line_thickness=3, color=predict_setting.color_table[labels[i]]
        )
    cv2.imshow('Detection result', img_origin)
    cv2.imwrite(predict_setting.save_img_path, img_origin)
    cv2.waitKey(0)
    return img_origin


def vedio_detect(input_video):
    vid = cv2.VideoCapture(input_video)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))
    video_writer = cv2.VideoWriter('video_result.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), video_fps, (video_width, video_height))

    start_time = time.time()
    with tf.variable_scope('yolov3'):
        yolov3 = YoloV3()
    yolov3_time = time.time()
    print("yolo v3 build time:%ss" % str(yolov3_time - start_time))

    for _ in range(video_frame_cnt):
        start_time = time.time()
        ret, img_origin = vid.read()
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

        print("bbox coords:%s, scores:%s, labels:%s" % (bboxes, scores, labels))

        for i in range(len(bboxes)):
            x0, y0, x1, y1 = bboxes[i]
            plot_one_box(
                img_origin, [x0, y0, x1, y1], label=predict_setting.classes[labels[i]] + ', {:.2f}%'.format(scores[i] * 100),
                line_thickness=3, color=predict_setting.color_table[labels[i]]
            )
        end_time = time.time()
        cv2.putText(
            img_origin, '%s ms' % str(int((end_time - start_time) * 1000)),
            (40, 40), 0, fontScale=1, color=(0, 255, 0), thickness=2
        )
        cv2.imshow('image', img_origin)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        video_writer.write(img_origin)
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
        img_detect(img_origin)
    elif args.file_type == 'vedio':
        vedio_detect(args.input_vedio)


if __name__ == '__main__':
    main()
