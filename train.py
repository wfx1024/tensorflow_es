# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import logging
from tqdm import trange
from setting import train_args
from utils.data_utils import create_iterator
from utils.misc_utils import make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms
from net.model import yolov3

# log
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S', filename=train_args.progress_log_path, filemode='w'
)


def get_learning_rate(global_step):
    """
    学习率
    :param global_step:
    :return:
    """
    if train_args.use_warm_up:
        learning_rate = tf.cond(
            tf.less(global_step, train_args.train_batch_num * train_args.warm_up_epoch),
            lambda: train_args.learning_rate_init * global_step / (train_args.train_batch_num * train_args.warm_up_epoch),
            lambda: config_learning_rate(train_args, global_step - train_args.train_batch_num * train_args.warm_up_epoch)
        )
    else:
        learning_rate = config_learning_rate(train_args, global_step)
    return learning_rate


def build_optimizer(learning_rate, loss, l2_loss, update_vars, global_step):
    """
    生成优化器
    :return:
    """
    optimizer = config_optimizer(train_args.optimizer_name, learning_rate)
    # BN操作
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # 梯度下降
        gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)  # 只优化update_vars中参数
        # 应用gradient clip, 防止梯度爆炸
        clip_grad_var = [gv if gv[0] is None else [
            tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)
    return train_op


def train():
    # dataset方法
    train_init_op, val_init_op, image_ids, image, y_true = create_iterator()

    # 是否训练placeholders
    is_training = tf.placeholder(tf.bool, name="phase_train")
    pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])

    # gpu nms 操作
    gpu_nms_op = gpu_nms(
        pred_boxes_flag, pred_scores_flag, train_args.class_num, train_args.nms_topk,
        train_args.score_threshold, train_args.nms_threshold
    )

    # 模型加载
    yolo_model = yolov3(train_args.class_num, train_args.anchors, train_args.use_label_smooth, train_args.use_focal_loss, train_args.batch_norm_decay, train_args.weight_decay, use_static_shape=False)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
    # 预测值
    y_pred = yolo_model.predict(pred_feature_maps)
    # loss
    loss = yolo_model.compute_loss(pred_feature_maps, y_true)
    l2_loss = tf.losses.get_regularization_loss()

    tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
    tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
    tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
    tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
    tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
    tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
    tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])

    # 加载除去yolov3/yolov3_head下Conv_6、Conv_14、Conv_22
    saver_to_restore = tf.train.Saver(
        var_list=tf.contrib.framework.get_variables_to_restore(
            include=train_args.restore_include, exclude=train_args.restore_exclude
        )
    )
    # 需要更新的变量
    update_vars = tf.contrib.framework.get_variables_to_restore(include=train_args.update_part)
    global_step = tf.Variable(
        float(train_args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]
    )

    # 学习率
    learning_rate = get_learning_rate(global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    # 是否要保存优化器的参数
    if not train_args.save_optimizer:
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()

    # 优化器
    train_op = build_optimizer(learning_rate, loss, l2_loss, update_vars, global_step)

    if train_args.save_optimizer:
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        print('\033[32m----------- Begin resotre weights  -----------')
        saver_to_restore.restore(sess, train_args.restore_path)
        print('\033[32m----------- Finish resotre weights  -----------')
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(train_args.log_dir, sess.graph)

        print('\n\033[32m----------- start to train -----------\n')
        best_mAP = -np.Inf

        for epoch in range(train_args.total_epoches):  # epoch
            print('\033[32m---------epoch:{}---------'.format(epoch))
            sess.run(train_init_op)  # 初始化训练集dataset
            # 初始化五种损失函数
            loss_total, loss_xy, loss_wh, loss_conf, loss_class\
                = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            for i in trange(train_args.train_batch_num):  # batch
                # print('\n\033[32m-----train batch:{}----'.format(i))
                # 优化器. summary, 预测值, gt, 损失, global_step, 学习率
                _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                    [train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                    feed_dict={is_training: True})
                # print('\033[32m-global_step:{}-'.format(__global_step))
                writer.add_summary(summary, global_step=__global_step)

                # 更新误差
                loss_total.update(__loss[0], len(__y_pred[0]))
                loss_xy.update(__loss[1], len(__y_pred[0]))
                loss_wh.update(__loss[2], len(__y_pred[0]))
                loss_conf.update(__loss[3], len(__y_pred[0]))
                loss_class.update(__loss[4], len(__y_pred[0]))

                # 验证集上验证
                if __global_step % train_args.train_evaluation_step == 0 and __global_step > 0:
                    # 召回率,精确率
                    recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred, __y_true, train_args.class_num, train_args.nms_threshold)

                    info = "epoch:{},global_step:{} | loss_total:{:.2f}, "\
                        .format(epoch, int(__global_step), loss_total.average)
                    info += "xy:{:.2f},wh:{:.2f},conf:{:.2f},class:{:.2f} | "\
                        .format(loss_xy.average, loss_wh.average, loss_conf.average, loss_class.average)
                    info += 'last batch:rec:{:.3f},prec:{:.3f} | lr:{:.5g}'\
                        .format(recall, precision, __lr)
                    print(info)

                    writer.add_summary(make_summary('evaluation/train_batch_recall', recall), global_step=__global_step)
                    writer.add_summary(make_summary('evaluation/train_batch_precision', precision), global_step=__global_step)

                    if np.isnan(loss_total.average):
                        raise ArithmeticError('梯度爆炸，修改参数后重新训练')

            # 保存模型
            if epoch % train_args.save_epoch == 0 and epoch > 0:
                if loss_total.average <= 2.:
                    print('\033[32m ----------- Begin sotre weights-----------')
                    print('\033[32m-loss_total.average{}'.format(loss_total.average))
                    saver_to_save.save(
                        sess,
                        train_args.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(
                            epoch, int(__global_step), loss_total.average, __lr
                        )
                    )
                    print('\033[32m ----------- Begin sotre weights  -----------')

            #  验证集评估评估方法
            if epoch % train_args.val_evaluation_epoch == 0 and epoch >= train_args.warm_up_epoch:  # 要过了warm up
                sess.run(val_init_op)

                val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
                    AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                val_preds = []

                for _ in trange(train_args.val_img_cnt):  # 在整个验证集上验证
                    __image_ids, __y_pred, __loss = sess.run(
                        [image_ids, y_pred, loss], feed_dict={is_training: False}
                    )
                    pred_content = get_preds_gpu(
                        sess, gpu_nms_op, pred_boxes_flag,
                        pred_scores_flag, __image_ids, __y_pred
                    )

                    val_preds.extend(pred_content)
                    # 更新误差
                    val_loss_total.update(__loss[0])
                    val_loss_xy.update(__loss[1])
                    val_loss_wh.update(__loss[2])
                    val_loss_conf.update(__loss[3])
                    val_loss_class.update(__loss[4])

                # 计算验证集mAP
                rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
                gt_dict = parse_gt_rec(train_args.val_file, train_args.img_size, train_args.letterbox_resize)

                info = 'Epoch: {}, global_step: {}, lr: {:.6g} \n'.format(epoch, __global_step, __lr)

                for j in range(train_args.class_num):
                    npos, nd, rec, prec, ap = voc_eval(
                        gt_dict, val_preds, j, iou_thres=train_args.eval_threshold, use_07_metric=train_args.use_voc_07_metric
                    )
                    info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(j, rec, prec, ap)
                    rec_total.update(rec, npos)
                    prec_total.update(prec, nd)
                    ap_total.update(ap, 1)

                mAP = ap_total.average
                info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'\
                    .format(rec_total.average, prec_total.average, mAP)
                info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'\
                    .format(val_loss_total.average, val_loss_xy.average,
                            val_loss_wh.average, val_loss_conf.average, val_loss_class.average)

                print(info)
                logging.info(info)

                if mAP > best_mAP:
                    best_mAP = mAP
                    saver_best.save(
                        sess,
                        train_args.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'
                        .format(epoch, int(__global_step), best_mAP, val_loss_total.average, __lr)
                    )

                writer.add_summary(make_summary('evaluation/val_mAP', mAP), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_recall', rec_total.average), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_precision', prec_total.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/total_loss', val_loss_total.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss_xy.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss_wh.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss_conf.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_class', val_loss_class.average), global_step=epoch)


if __name__ == '__main__':
    train()
