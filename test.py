# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
from model import yolov3


def vedio_detect(input_args):
    input_args.anchors = parse_anchors(input_args.anchor_path)
    input_args.classes = read_class_names(input_args.class_name_path)
    input_args.num_class = len(input_args.classes)

    color_table = get_color_table(input_args.num_class)

    vid = cv2.VideoCapture(input_args.input_video)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))

    if input_args.save_video:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, input_args.new_size[1], input_args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(input_args.num_class, input_args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, input_args.num_class, max_boxes=200, score_thresh=0.3,
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, input_args.restore_path)

        for i in range(video_frame_cnt):
            ret, img_ori = vid.read()
            if input_args.letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, input_args.new_size[0], input_args.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, tuple(input_args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            start_time = time.time()
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            end_time = time.time()

            # rescale the coordinates to the original image
            if input_args.letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(input_args.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(input_args.new_size[1]))

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(img_ori, [x0, y0, x1, y1],
                             label=input_args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                             color=color_table[labels_[i]])
            cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imshow('image', img_ori)
            if input_args.save_video:
                videoWriter.write(img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        if input_args.save_video:
            videoWriter.release()


def img_detect(input_args):
    # 解析输入
    input_args.anchors = parse_anchors(input_args.anchor_path)
    input_args.classes = read_class_names(input_args.class_name_path)
    input_args.num_class = len(input_args.classes)
    color_table = get_color_table(input_args.num_class)
    img_ori = cv2.imread(input_args.input_image)

    if input_args.letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, input_args.new_size[0], input_args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(input_args.new_size))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, input_args.new_size[1], input_args.new_size[0], 3], name='input_data')
        with tf.variable_scope('yolov3'):
            yolo_model = yolov3(input_args.num_class, input_args.anchors)
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, input_args.num_class, max_boxes=200, score_thresh=0.3,
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, input_args.restore_path)

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        if input_args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(input_args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(input_args.new_size[1]))

        print("box coords:", boxes_, '\n' + '*' * 30)
        print("scores:", scores_, '\n' + '*' * 30)
        print("labels:", labels_)

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(
                img_ori, [x0, y0, x1, y1],
                label=input_args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                color=color_table[labels_[i]]
            )
        cv2.imshow('Detection result', img_ori)
        cv2.imwrite('detection_result.jpg', img_ori)
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description="YOLO V3 检测文件")
    parser.add_argument("--detect_object", default="img", type=str, help="检测目标-img或vedio")
    parser.add_argument("--input_image", default="./data/demo_data/dog.jpg", type=str, help="图片路径")
    parser.add_argument("--input_video", type=str, help="视频路径")
    parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt", help="anchor 文件路径")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416], help="图片改变后的大小")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否使用letterbox")
    parser.add_argument("--class_name_path", type=str, default="./data/coco.names", help="类别文件路径")
    parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt", help="weights路径")
    parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否保存视频")

    input_args = parser.parse_args()
    # 图片检测
    if input_args.detect_object == 'img':
        img_origin = cv2.imread(input_args.input_image)  # 原始图片
        if img_origin is None:
            raise Exception("未找到图片文件！")
        img_detect(input_args)

    # 视频检测
    elif input_args.detect_object == 'vedio':
        vid = cv2.VideoCapture(input_args.input_vedio)
        if vid is None:
            raise Exception("未找到视频文件！")
        vedio_detect(input_args)


if __name__ == '__main__':
    main()
