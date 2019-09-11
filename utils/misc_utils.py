# coding: utf-8

import numpy as np
import tensorflow as tf
import random

from tensorflow.core.framework import summary_pb2


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)


def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def shuffle_and_overwrite(file_name):
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)


def update_dict(ori_dict, new_dict):
    if not ori_dict:
        return new_dict
    for key in ori_dict:
        ori_dict[key] += new_dict[key]
    return ori_dict


def list_add(ori_list, new_list):
    for i in range(len(ori_list)):
        ori_list[i] += new_list[i]
    return ori_list


def config_learning_rate(args, global_step):
    """
    获取学习率
    :param args:
    :param global_step:
    :return:
    """
    # 指数
    if args.lr_type == 'exponential':
        lr_tmp = tf.train.exponential_decay(
            args.learning_rate_init, global_step, args.lr_decay_freq,
            args.lr_decay_factor, staircase=True, name='exponential_learning_rate'
        )
        return tf.maximum(lr_tmp, args.lr_lower_bound)

    # 余弦
    elif args.lr_type == 'cosine_decay':
        train_steps = (args.total_epoches - float(args.use_warm_up) * args.warm_up_epoch) * args.train_batch_num
        return args.lr_lower_bound + 0.5 * (args.learning_rate_init - args.lr_lower_bound) * \
            (1 + tf.cos(global_step / train_steps * np.pi))

    elif args.lr_type == 'cosine_decay_restart':
        return tf.train.cosine_decay_restarts(
            args.learning_rate_init, global_step,
            args.lr_decay_freq, t_mul=2.0, m_mul=1.0,
            name='cosine_decay_learning_rate_restart'
        )

    elif args.lr_type == 'fixed':
        return tf.convert_to_tensor(args.learning_rate_init, name='fixed_learning_rate')

    elif args.lr_type == 'piecewise':
        return tf.train.piecewise_constant(
            global_step, boundaries=args.pw_boundaries,
            values=args.pw_values, name='piecewise_learning_rate'
        )
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    """
    优化器选择
    :param optimizer_name:
    :param learning_rate:
    :param decay:
    :param momentum:
    :return:
    """
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')
