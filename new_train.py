# coding: utf-8

from utils.dataset_tricks import build_train_dataset, build_val_dataset, create_iterator
import tensorflow as tf
from net.yolov3 import YoloV3

def main():
    with tf.variable_scope('yolov3'):
        yolov3 = YoloV3(is_training=True)
    yolov3.train()


if __name__ == '__main__':
    main()

