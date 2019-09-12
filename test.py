# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import setting.predict_args as pred_args
from utils.nms_utils import gpu_nms
from utils.plot_utils import plot_one_box
from utils.data_aug import letterbox_resize
from net.model import yolov3


def img_detect(input_args):
    """
    图片检测
    :param input_args:
    :return:
    """
    img_ori = cv2.imread(input_args.input_image)  # opencv 打开
    if input_args.use_letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, pred_args.new_size[0], pred_args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(pred_args.new_size))

    # img 转RGB, 转float, 归一化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    sess = tf.Session()

    input_data = tf.placeholder(
        tf.float32, [1, pred_args.new_size[1], pred_args.new_size[0], 3], name='input_data'
    )
    with tf.variable_scope('yolov3'):
        yolo_model = yolov3(pred_args.num_class, pred_args.anchors)
        pred_feature_maps = yolo_model.forward(input_data, False)

    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(
        pred_boxes, pred_scores, pred_args.num_class,
        max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, pred_args.weight_path)

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # 还原坐标到原图
    if input_args.use_letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori / float(pred_args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori / float(pred_args.new_size[1]))

    print('box coords:', boxes_, '\n' + '*' * 30)
    print('scores:', scores_, '\n' + '*' * 30)
    print('labels:', labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(
            img_ori, [x0, y0, x1, y1],
            label=pred_args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
            color=pred_args.color_table[labels_[i]]
        )
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite(pred_args.output_image, img_ori)
    cv2.waitKey(0)
    sess.close()


def video_detect(input_args):
    vid = cv2.VideoCapture(input_args.input_video)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(pred_args.output_video, fourcc, video_fps, (video_width, video_height))

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, pred_args.new_size[1], pred_args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(pred_args.num_class, pred_args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)

        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
        pred_scores = pred_confs * pred_probs
        boxes, scores, labels = gpu_nms(
            pred_boxes, pred_scores, pred_args.num_class,
            max_boxes=200, score_thresh=0.3, nms_thresh=0.45
        )
        saver = tf.train.Saver()
        saver.restore(sess, pred_args.weight_path)

        for i in range(video_frame_cnt):
            ret, img_ori = vid.read()
            if input_args.use_letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, pred_args.new_size[0], pred_args.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, tuple(pred_args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            start_time = time.time()
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            end_time = time.time()

            if input_args.use_letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(pred_args.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(pred_args.new_size[1]))

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(img_ori, [x0, y0, x1, y1],
                             label=pred_args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                             color=pred_args.color_table[labels_[i]])
            cv2.putText(
                img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000),
                (40, 40), 0, fontScale=1, color=(0, 255, 0), thickness=2
            )
            cv2.imshow('Detection result', img_ori)
            video_writer.write(img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        video_writer.release()


def main():
    parser = argparse.ArgumentParser(description='YOLO V3 检测文件')
    parser.add_argument('--detect_object', default=pred_args.detect_object, type=str, help='检测目标-img或video')
    parser.add_argument('--input_image', default=pred_args.input_image, type=str, help='图片路径')
    parser.add_argument('--input_video', default=pred_args.input_video, type=str, help='视频路径')
    parser.add_argument('--use_letterbox_resize', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否使用letterbox')

    input_args = parser.parse_args()
    # 图片检测
    if input_args.detect_object == 'img':
        img_origin = cv2.imread(input_args.input_image)  # 原始图片
        if img_origin is None:
            raise Exception('未找到图片文件！')
        img_detect(input_args)

    # 视频检测
    elif input_args.detect_object == 'video':
        vid = cv2.VideoCapture(input_args.input_video)
        if vid is None:
            raise Exception('未找到视频文件!')
        video_detect(input_args)


if __name__ == '__main__':
    main()
