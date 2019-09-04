# -*- coding:utf-8 -*-

import tensorflow as tf


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
