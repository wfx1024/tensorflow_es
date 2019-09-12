# -*- coding:utf-8 -*-

import tensorflow as tf
import cv2
from tqdm import *
import time
from utils.data_utils import create_iterator


def test_dim_size():
    # slim = tf.contrib.slim
    input = tf.Variable(tf.random_uniform([1, 5, 5, 3]))
    kernel1 = tf.Variable(tf.random_uniform([1, 1, 3, 1]))  # 1,1,3,1
    kernel2 = tf.concat([kernel1, kernel1], 3)  # 1,1,3,3
    out1 = tf.nn.conv2d(input, kernel1, strides=[1, 1, 1, 1], padding='VALID')
    out2 = tf.nn.conv2d(input, kernel2, strides=[1, 1, 1, 1], padding='VALID')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("\ninput----->\n", input.eval())
        print("\nkernel----->\n", kernel1.eval())
        print("\nkernel2----->\n", kernel2.eval())
        print("\n----->\n", out1.eval())
        print("\n----->\n", out2.eval())


def test_dataset():
    train_init_op, val_init_op, image_ids, image, y_true = create_iterator()
    with tf.Session() as sess:
        sess.run(train_init_op)
        for i in range(2):
            sess.run(image)


def test_txt_write():
    first = []
    second = []
    f = open('mergeTXT.txt', 'w')
    with open('first.txt', 'r') as f1:
        for line in f1:
            line = line.strip()
            first.append(line)
    with open('second.txt', 'r') as f2:
        for line2 in f2:
            line2 = line2.strip()
            second.append(line2)
    for i in range(0, 399):
        result = first[i] + '\t' + second[i] + '\n'
        f.write(result)


def test_tqdm():
    with tqdm(total=100) as pbar:
        for i in range(10):
            time.sleep(1)
            pbar.update(10)


def test_plot_bbox():
    img = cv2.imread('data/demo_data/dog.jpg')
    # 
    cv2.rectangle(img, (10, 100), (20, 200), (0, 255, 0), 2)
    cv2.imshow('img_detect', img)
    cv2.waitKey(0)


def test_watch_save_weights():
    from tensorflow.python import pywrap_tensorflow
    # model_dir = 'checkpoint/model-epoch_12_step_64_loss_1.9270_lr_0.0001'
    model_dir = 'data/darknet_weights/yolov3.ckpt'
    reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("have {} tensor".format(len(var_to_shape_map)))
    for key in var_to_shape_map:
        print("tensor_name:{}, shape:{}".format(key, reader.get_tensor(key).shape))
