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


def test_tf_data():
    """
       验证数据
       :return:
       """
    import args
    from utils.data_utils import get_batch_data
    train_dataset = tf.data.TextLineDataset(args.train_file)
    train_dataset = train_dataset.shuffle(args.train_img_cnt)  # 先随机重排
    train_dataset = train_dataset.batch(args.batch_size)  # 分批
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(
            get_batch_data,
            inp=[x, args.class_num, args.img_size, args.anchors, 'train',
                 args.multi_scale_train, args.use_mix_up, args.letterbox_resize],
            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    # train_dataset = train_dataset.prefetch(args.prefetech_buffer)  # 每次取5
    iterator = train_dataset.make_one_shot_iterator()  # 构建迭代器，自动初始化
    next_element = iterator.get_next()  # 获取下一个元素op
    with tf.Session() as sess:
        for i in range(12):  # 获取每一个元素
            value = sess.run(next_element)
            print(i, "--->", value)
    return train_dataset



