# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def conv2d(inputs, filters, kernel_size, strides=1):
    """
    net = conv2d(net, 64,  3, strides=2)
    卷积构建
    :param inputs: 输入
    :param filters: 卷积核数量（也就是输出的channels）
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :return:
    """
    def _fixed_padding(inputs, kernel_size):
        """
        固定填充,不依赖输入大小
        :param inputs:
        :param kernel_size:
        :return:
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(
            inputs,
            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]],
            mode='CONSTANT'
        )
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)

    inputs = slim.conv2d(
        inputs, filters, kernel_size, stride=strides,
        padding=('SAME' if strides == 1 else 'VALID')
    )
    return inputs


def res_block(inputs, filters):
    """
    残差块
    :param inputs: 输入
    :param filters: 过滤器数量
    :return:
    """
    shortcut = inputs  # 输入副本
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = net + shortcut   # 拼接
    return net


def darknet53_body(inputs):
    """
    darknet53网络主体，它包含53个卷积层
    每个卷后面都有BN层和Leaky ReLU激活层。
    下采样由带有stride = 2的conv层完成。
    :param inputs: input
    :return:
    """
    
    # 第一层的两个卷积
    net = conv2d(inputs, 32,  3, strides=1)
    net = conv2d(net, 64,  3, strides=2)

    # res_block * 1
    net = res_block(net, 32)
    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3


def yolo_block(inputs, filters):
    """
    检测模块
    由上采样层、3个具有线性激活功能的卷积层，从而在3种不同的尺度上进行检测。
    :param inputs:
    :param filters:
    :return:
    """
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    # 使用近邻值插入调整图像
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs


