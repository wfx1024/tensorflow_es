# coding: utf-8
import tensorflow as tf
from data_utils import get_batch_data
import args as args


def build_train_dataset():
    """
    构建验证数据
    :return:
    """
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
    train_dataset = train_dataset.prefetch(args.prefetech_buffer)  # 每次取1
    return train_dataset


def build_val_dataset():
    """
    构建验证数据集
    :return:
    """
    val_dataset = tf.data.TextLineDataset(args.val_file)
    val_dataset = val_dataset.batch(1)  # 一批一个
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(
            get_batch_data,
            inp=[x, args.class_num, args.img_size, args.anchors,
                 'val', False, False, args.letterbox_resize],
            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    val_dataset = val_dataset.prefetch(args.prefetech_buffer)
    return val_dataset


def create_iterator():
    """
    创建迭代器
    :return:
    """
    print('\n----------- Begin building dataset  -----------\n')
    train_dataset = build_train_dataset()  # 训练集
    val_dataset = build_val_dataset()  # 验证集
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    # 获得一条数据
    image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    # 如果丢失了shape,则手动设置
    image_ids.set_shape([None])
    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])
    print('\n----------- Finish building dataset  -----------\n')
    return train_init_op, val_init_op, image_ids, image, y_true
