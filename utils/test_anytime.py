# -*- coding:utf-8 -*-

import tensorflow as tf
import random
import os
import numpy as np


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


def test_tf_data():
    """
    验证数据
    :return:
    """
    training_dataset = tf.data.Dataset.range(100).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    handle = tf.placeholder(tf.string, shape=[])  # placeholder作为参数
    # 定义一个可让您在两个数据集之间切换的可馈送迭代器，
    iterator = tf.data.Iterator.from_string_handle(
        handle,
        training_dataset.output_types,
        training_dataset.output_shapes
    )
    next_element = iterator.get_next()
    # 定义两个迭代器
    training_iterator = training_dataset.make_one_shot_iterator()  # make_one_shot_iterator自动init
    validation_iterator = validation_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        # string_handle()获得该iterator的handle参数
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        for _ in range(20):  # 训练集上运行20个epochs
            for i in range(200):
                # 将各自的handle馈送回去
                value = sess.run(next_element, feed_dict={handle: training_handle})
                print("train:", i, "-->", value)
            sess.run(validation_iterator.initializer)
            for i in range(50):
                sess.run(next_element, feed_dict={handle: validation_handle})
                print("test:", i, "-->", value)


def test_color():
    random.seed(2)
    color_table = {}
    for i in range(20):
        rgb = [0, 0, 0]
        while rgb[0] < 100 and rgb[1] < 100 and rgb[2] < 100:
            rgb = [random.randint(0, 255) for _ in range(3)]
        color_table[i] = rgb
    print(color_table)


def test_ckpt_variable():
    """
    查看保存的所有信息
    :return:
    """
    from tensorflow.python import pywrap_tensorflow
    model_dir = '../data/darknet_weights'
    checkpoint_path = os.path.join(model_dir, "yolov3.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))


def test_dataset():
    from utils.dataset_tricks import build_train_dataset
    train_dataset = build_train_dataset()  # 训练集
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(train_dataset)
    with tf.Session() as sess:
        sess.run(training_init_op)
        for _ in range(1):
            result = sess.run(next_element)
            print(result)


def test_data_set():
    features = np.array([1, 2, 3, 4])
    labels = np.array([10, 20, 30, 40])
    training_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    training_dataset = training_dataset.map(
        lambda x, y: (x + 1, y + 2)
    )

    iterator = tf.data.Iterator.from_structure(
        training_dataset.output_types,
        training_dataset.output_shapes
    )
    a, b = iterator.get_next()
    # 初始化迭代器
    training_init_op = iterator.make_initializer(training_dataset)
    with tf.Session() as sess:
        for epoch in range(20):  # 训练集上运行20个epochs
            sess.run(training_init_op)
            for i in range(4):
                value = sess.run(a)
                print(epoch, "-->", i, "-->", value)
