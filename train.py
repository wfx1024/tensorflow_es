# coding: utf-8

from utils.dataset_tricks import build_train_dataset, build_val_dataset, create_iterator
import tensorflow as tf
from net.yolov3 import YoloV3
import config.setting as setting
import config.train_setting as train_setting
from utils.training_utils import AverageMeter, get_learning_rate, config_optimizer
from tqdm import trange
from utils.nms import gpu_nms
from utils.eval_utils import get_preds_gpu, evaluate_on_gpu, parse_gt_rec, voc_eval
from tensorflow.core.framework import summary_pb2
import numpy as np


is_training = True
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
score_threshold = setting.get_score_threshold(is_training=True)
gpu_nms_op = gpu_nms(
    pred_boxes_flag, pred_scores_flag, setting.class_num, train_setting.nms_topk,
    score_threshold, setting.nms_threshold
)
global_step = tf.Variable(
    float(train_setting.global_step),
    trainable=False,
    collections=[tf.GraphKeys.LOCAL_VARIABLES]
)


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def evaluate_each_batch(
        loss_total, loss_xy, loss_wh, loss_conf, loss_class,
        writer, epoch, summary, y_pred, y_true, loss, global_step, learning_rate):
    """
    每个batch的评估方法
    :param writer:
    :param epoch:
    :param summary:
    :param y_pred:
    :param y_true:
    :param loss:
    :param global_step:
    :param learning_rate:
    :return:
    """
    score_threshold = setting.get_score_threshold(is_training=True)

    # 注册GPU nms, 方便后续
    gpu_nms_op = gpu_nms(
        pred_boxes_flag, pred_scores_flag, setting.class_num,
        train_setting.nms_topk, score_threshold, setting.nms_threshold
    )
    writer.add_summary(summary, global_step=global_step)
    # 更新误差
    loss_total.update(loss[0], len(y_pred[0]))
    loss_xy.update(loss[1], len(y_pred[0]))
    loss_wh.update(loss[2], len(y_pred[0]))
    loss_conf.update(loss[3], len(y_pred[0]))
    loss_class.update(loss[4], len(y_pred[0]))

    if global_step % train_setting.train_evaluation_step == 0 and global_step > 0:
        # 召回率,精确率
        recall, precision = evaluate_on_gpu(
            sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag,
            y_pred, y_true, setting.class_num, setting.nms_threshold
        )
        info = "Epoch:{}, global_step: {} | loss: total: {:.2f}, " \
            .format(epoch, int(global_step), loss_total.average)
        info += "xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | " \
            .format(loss_xy.average, loss_wh.average, loss_conf.average, loss_class.average)
        info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, learning_rate)
        print(info)

        writer.add_summary(
            make_summary('evaluation/train_batch_recall', recall),
            global_step=global_step
        )
        writer.add_summary(
            make_summary('evaluation/train_batch_precision', precision),
            global_step=global_step
        )

        if np.isnan(loss_total.average):
            raise ArithmeticError('****' * 10 + '\n梯度爆炸，修改参数后重新训练')


def each_batch(sess, saver_to_restore, image, y_true):
    # weights
    sess.run(tf.initialize_all_variables())
    saver_to_restore.restore(sess, setting.weights_path)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(train_setting.log_dir, sess.graph)

    # dataset 得到image, y_true
    image, y_true, summary, global_step = sess.run(
        image, y_true, merged, global_step
    )

    # 学习率
    learning_rate = get_learning_rate(global_step)
    learning_rate = sess.run(learning_rate)

    # 反向传播
    _, y_pred, loss = yolov3.sess.run(
        [yolov3.train_op, yolov3.y_pred, yolov3.loss],
        feed_dict={yolov3.input_data: image}
    )

    score_threshold = setting.get_score_threshold(is_training=True)

    # 注册GPU nms, 方便后续
    gpu_nms_op = gpu_nms(
        pred_boxes_flag, pred_scores_flag, setting.class_num,
        train_setting.nms_topk, score_threshold, setting.nms_threshold
    )
    writer.add_summary(summary, global_step=global_step)
    # 更新误差
    loss_total.update(loss[0], len(y_pred[0]))
    loss_xy.update(loss[1], len(y_pred[0]))
    loss_wh.update(loss[2], len(y_pred[0]))
    loss_conf.update(loss[3], len(y_pred[0]))
    loss_class.update(loss[4], len(y_pred[0]))

    if global_step % train_setting.train_evaluation_step == 0 and global_step > 0:
        # 召回率,精确率
        recall, precision = evaluate_on_gpu(
            sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag,
            y_pred, y_true, setting.class_num, setting.nms_threshold
        )
        info = "Epoch:{}, global_step: {} | loss: total: {:.2f}, " \
            .format(epoch, int(global_step), loss_total.average)
        info += "xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | " \
            .format(loss_xy.average, loss_wh.average, loss_conf.average, loss_class.average)
        info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, learning_rate)
        print(info)

        writer.add_summary(
            make_summary('evaluation/train_batch_recall', recall),
            global_step=global_step
        )
        writer.add_summary(
            make_summary('evaluation/train_batch_precision', precision),
            global_step=global_step
        )

        if np.isnan(loss_total.average):
            raise ArithmeticError('****' * 10 + '\n梯度爆炸，修改参数后重新训练')


def train(self):
    """
    训练
    :return:
    """
    # dataset
    train_init_op, val_init_op, image_ids, image, y_true = create_iterator()

    # YOLO V3
    with tf.variable_scope('yolov3'):
        yolov3 = YoloV3(self.is_training)

    # saver
    saver_to_restore = tf.train.Saver(
        var_list=tf.contrib.framework.get_variables_to_restore(
            include=train_setting.restore_include, exclude=train_setting.restore_exclude
        )
    )
    update_vars = tf.contrib.framework.get_variables_to_restore(include=train_setting.update_part)

    print('\n----------- start to train -----------\n')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver_to_restore.restore(sess, setting.weights_path)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(train_setting.log_dir, sess.graph)

        for epoch in range(train_setting.total_epoches):  # 训练100epoch
            sess.run(train_init_op)
            # 初始化五种损失函数
            loss_total = AverageMeter()
            loss_xy = AverageMeter()
            loss_wh = AverageMeter()
            loss_conf = AverageMeter()
            loss_class = AverageMeter()

            for _ in trange(train_setting.train_batch_num):  # batch
                each_batch(
                    loss_total, loss_xy, loss_wh, loss_conf, loss_class,
                    writer, epoch, summary, y_pred, y_true, loss, global_step, learning_rate
                )

            _save_weight(epoch, global_step, learning_rate)
            _evaluate_in_val(epoch, global_step, learning_rate)


def _save_weight(self, epoch, global_step, learning_rate):
    """
    保存权重
    :param epoch: epoch index
    :param global_step: 步数
    :param learning_rate: 学习率
    :return:
    """
    if epoch % train_setting.save_epoch == 0 and epoch > 0:
        if self.loss_total.average <= 2.:
            self.saver_to_save.save(
                self.sess,
                train_setting.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'
                .format(epoch, int(global_step), self.loss_total.average, learning_rate)
            )


def _evaluate_in_val(self, epoch, __global_step, __lr):
    """
    验证集评估评估方法
    :param epoch:
    :param __global_step:
    :param __lr:
    :return:
    """
    if epoch % train_setting.val_evaluation_epoch == 0 and epoch >= train_setting.warm_up_epoch:
        self.sess.run(self.val_init_op)

        # 初始化五种代价函数
        val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        val_preds = []

        for _ in trange(train_setting.val_img_cnt):
            __image_ids, __y_pred, __loss = self.sess.run(
                [self.image_ids, self.y_pred, self.loss],
                feed_dict={self.input_data: self.image}
            )
            pred_content = get_preds_gpu(
                self.sess, self.gpu_nms_op, self.pred_scores_flag, __image_ids, __y_pred
            )
            val_preds.extend(pred_content)
            val_loss_total.update(__loss[0])
            val_loss_xy.update(__loss[1])
            val_loss_wh.update(__loss[2])
            val_loss_conf.update(__loss[3])
            val_loss_class.update(__loss[4])

        # 计算 mAP
        rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
        gt_dict = parse_gt_rec(train_setting.val_file, setting.img_size, setting.letterbox_resize_used)

        info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)

        for ii in range(setting.class_num):
            npos, nd, rec, prec, ap = voc_eval(
                gt_dict, val_preds, ii, iou_thres=train_setting.eval_threshold,
                use_07_metric=train_setting.use_voc_07_metric
            )
            info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
            rec_total.update(rec, npos)
            prec_total.update(prec, nd)
            ap_total.update(ap, 1)

        mAP = ap_total.average
        info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n' \
            .format(rec_total.average, prec_total.average, mAP)

        info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n' \
            .format(val_loss_total.average, val_loss_xy.average, val_loss_wh.average,
                    val_loss_conf.average, val_loss_class.average)

        print(info)

        if mAP > self.best_mAP:
            self.best_mAP = mAP
            self.saver_best.save(
                self.sess,
                train_setting.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'
                .format(epoch, int(__global_step), self.best_mAP, val_loss_total.average, __lr)
            )
        writer.add_summary(
            make_summary('evaluation/val_mAP', mAP),
            global_step=epoch
        )
        writer.add_summary(
            make_summary('evaluation/val_recall', rec_total.average),
            global_step=epoch
        )
        writer.add_summary(make_summary(
            'evaluation/val_precision', prec_total.average),
            global_step=epoch
        )
        writer.add_summary(
            make_summary('validation_statistics/total_loss', val_loss_total.average),
            global_step=epoch
        )
        writer.add_summary(
            make_summary('validation_statistics/loss_xy', val_loss_xy.average),
            global_step=epoch
        )
        writer.add_summary(
            make_summary('validation_statistics/loss_wh', val_loss_wh.average),
            global_step=epoch
        )
        writer.add_summary(
            make_summary('validation_statistics/loss_conf', val_loss_conf.average),
            global_step=epoch)
        writer.add_summary(
            make_summary('validation_statistics/loss_class', val_loss_class.average),
            global_step=epoch)


def main():
    train()


if __name__ == '__main__':
    main()
