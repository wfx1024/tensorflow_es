# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from tqdm import trange
from setting import train_args
from utils.data_utils import create_iterator
from utils.misc_utils import make_summary, config_learning_rate, config_optimizer, AverageMeter, Loss5
from utils.eval_utils import evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms
from net.model import yolov3


"""
train or fine-tuning
"""


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
    print('\033[32m----------- Begin building optimizer  -----------\033[0m')
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
    print('\033[32m----------- Finish building optimizer  -----------\033[0m')
    return train_op


class Train:
    def __init__(self):
        # 是否训练placeholders
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
        self.pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
        self.best_mAP = -np.Inf
        self.l2_loss = tf.losses.get_regularization_loss()
        self.global_step = tf.Variable(
            float(train_args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]
        )
        # dataset方法
        self.train_init_op, self.val_init_op, self.image_ids, self.image, self.y_true = create_iterator()

        self.sess = tf.Session()
        self.epoch = 0
        self.writer = tf.summary.FileWriter(train_args.log_dir, self.sess.graph)
        # 训练集5种损失
        self.loss_5 = Loss5()
        self.__pre_operate()

    def __loss_summary(self):
        tf.summary.scalar('train_batch_statistics/total_loss', self.loss[0])
        tf.summary.scalar('train_batch_statistics/loss_xy', self.loss[1])
        tf.summary.scalar('train_batch_statistics/loss_wh', self.loss[2])
        tf.summary.scalar('train_batch_statistics/loss_conf', self.loss[3])
        tf.summary.scalar('train_batch_statistics/loss_class', self.loss[4])
        tf.summary.scalar('train_batch_statistics/loss_l2', self.l2_loss)
        tf.summary.scalar('train_batch_statistics/loss_ratio', self.l2_loss / self.loss[0])
        tf.summary.scalar('learning_rate', self.learning_rate)

    def __pre_operate(self):
        """
        初始化部分操作
        :return:
        """
        # gpu nms 操作
        self.gpu_nms_op = gpu_nms(
            self.pred_boxes_flag, self.pred_scores_flag, train_args.class_num, train_args.nms_topk,
            train_args.score_threshold, train_args.nms_threshold
        )

        # 模型加载
        yolo_model = yolov3(
            train_args.class_num, train_args.anchors, train_args.use_label_smooth, train_args.use_focal_loss,
            train_args.batch_norm_decay, train_args.weight_decay, use_static_shape=False
        )

        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(self.image, is_training=self.is_training)

        # 预测值
        self.y_pred = yolo_model.predict(pred_feature_maps)
        # loss
        self.loss = yolo_model.compute_loss(pred_feature_maps, self.y_true)
        # 学习率
        self.learning_rate = get_learning_rate(self.global_step)
        self.__loss_summary()

        # 加载Saver
        self.saver_to_restore = tf.train.Saver(
            var_list=tf.contrib.framework.get_variables_to_restore(
                include=train_args.restore_include, exclude=train_args.restore_exclude
            )
        )

        # 是否要保存优化器的参数
        if not train_args.save_optimizer:
            self.saver_to_save = tf.train.Saver()
            self.saver_best = tf.train.Saver()

        # 需要更新的变量
        self.update_vars = tf.contrib.framework.get_variables_to_restore(include=train_args.update_part)
        # 优化器
        self.train_op = build_optimizer(self.learning_rate, self.loss, self.l2_loss, self.update_vars, self.global_step)

        if train_args.save_optimizer:
            self.saver_to_save = tf.train.Saver()
            self.saver_best = tf.train.Saver()

        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        print('\033[32m----------- Begin resotre weights  -----------\033[0m')
        self.saver_to_restore.restore(self.sess, train_args.restore_path)
        print('\033[32m----------- Finish resotre weights  -----------\033[0m')
        self.merged = tf.summary.merge_all()

    def train(self):
        print('\n\033[32m-----------Begin train -----------\033[0m\n')
        for epoch in range(train_args.total_epoches):  # epoch
            print('\033[32m---------epoch:{}---------\033[0m'.format(epoch))
            self.epoch = epoch
            self.sess.run(self.train_init_op)  # 初始化训练集dataset

            for _ in trange(train_args.train_batch_num):  # batch
                # 优化器. summary, 预测值, gt, 损失, global_step, 学习率
                _, __image_ids, summary, __y_pred, __y_true, __loss, __global_step, __lr = self.sess.run(
                    [self.train_op, self.image_ids, self.merged, self.y_pred,
                     self.y_true, self.loss, self.global_step, self.learning_rate],
                    feed_dict={self.is_training: True}
                )
                self.writer.add_summary(summary, global_step=__global_step)

                # 更新误差 loss_total, loss_xy, loss_wh, loss_conf, loss_class
                self.loss_5.update(__loss, len(__y_pred[0]))

            if __global_step % train_args.train_evaluation_step == 0 and __global_step > 0:
                self.__evaluate(__y_pred, __y_true, __global_step, __lr)

            # 保存模型
            if epoch % train_args.save_epoch == 0 and epoch > 0:
                if self.loss_5.loss_total.average <= 2.:
                    print('\033[32m ----------- Begin sotre weights-----------\033[0m')
                    self.saver_to_save.save(
                        self.sess,
                        train_args.save_dir + 'model_epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(
                            epoch, int(__global_step), self.loss_5.loss_total.average, __lr
                        )
                    )
                    print('\033[32m ----------Finish sotre weights-----------\033[0m')
            if epoch % train_args.val_evaluation_epoch == 0 and epoch >= train_args.warm_up_epoch:  # 要过了warm up
                self.__evaluate_in_val(__y_pred, __y_true, __global_step)
        print('\n\033[32m-----------Finish training -----------\033[0m\n')

    def __evaluate(self, __y_pred, __y_true, __global_step, __lr):
        """
        验证
        :return:
        """
        print('\033[32m -----------Begin evaluating-----------\033[0m')
        # 召回率,精确率
        recall, precision = evaluate_on_gpu(
            self.sess, self.gpu_nms_op, self.pred_boxes_flag, self.pred_scores_flag,
            __y_pred, __y_true, train_args.class_num, train_args.nms_threshold)

        info = '\nepoch:{}, global step{} ||  '.format(self.epoch, int(__global_step))
        info += 'loss_total:{:.2f}, '.format(self.loss_5.loss_total.average)
        info += 'loss_xy:{:.2f}, '.format(self.loss_5.loss_xy.average)
        info += 'loss_wh:{:.2f}, '.format(self.loss_5.loss_wh.average)
        info += 'loss_conf:{:.2f}, '.format(self.loss_5.loss_conf.average)
        info += 'loss_class:{:.2f} || '.format(self.loss_5.loss_class.average)
        info += '\nlast batch-->rec:{:.3f}, precision:{:.3f} | learning rate:{:.5g}' .format(recall, precision, __lr)
        print(info)

        self.writer.add_summary(
            make_summary('evaluation/train_batch_recall', recall), global_step=__global_step
        )
        self.writer.add_summary(
            make_summary('evaluation/train_batch_precision', precision), global_step=__global_step
        )

        if np.isnan(self.loss[0].average):
            raise ArithmeticError('梯度爆炸，修改参数后重新训练')
        print('\033[32m -----------Finish evaluating-----------\033[0m')

    def __evaluate_in_val(self, __global_step, __lr):
        """
        验证集评估评估方法
        :param __global_step:
        :param __lr:
        :return:
        """
        print('\033[32m -----Begin evaluating in val data-----------\033[0m')
        self.sess.run(self.val_init_op)
        val_loss_5 = Loss5()
        val_preds = []
        for i in trange(train_args.val_img_cnt):  # 在整个验证集上验证
            __image_ids, __y_pred, __loss = self.sess.run(
                [self.image_ids, self.y_pred, self.loss], feed_dict={self.is_training: False}
            )
            pred_content = get_preds_gpu(
                self.sess, self.gpu_nms_op, self.pred_boxes_flag,
                self.pred_scores_flag, __image_ids, __y_pred
            )

            val_preds.extend(pred_content)
            # 更新训练集误差
            val_loss_5.update(__loss)
            if i % 300 == 0:
                print(i, "--loss-->", __loss)

        # 计算验证集mAP
        rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
        gt_dict = parse_gt_rec(train_args.val_file, train_args.img_size, train_args.letterbox_resize)

        print('\033[32m -----Begin calculate mAP-------\033[0m')
        info = 'Epoch: {}, global_step: {}, lr: {:.6g} \n'.format(self.epoch, __global_step, __lr)  # todo
        for j in range(train_args.class_num):
            npos, nd, rec, prec, ap = voc_eval(
                gt_dict, val_preds, j, iou_thres=train_args.eval_threshold,
                use_07_metric=train_args.use_voc_07_metric
            )
            info += 'eval: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(j, rec, prec, ap)
            rec_total.update(rec, npos)
            prec_total.update(prec, nd)
            ap_total.update(ap, 1)

        mAP = ap_total.average
        info += 'eval: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n' \
            .format(rec_total.average, prec_total.average, mAP)
        info += 'eval: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'\
            .format(
                val_loss_5.loss_total.average,
                val_loss_5.loss_xy.average,
                val_loss_5.loss_wh.average,
                val_loss_5.loss_conf.average,
                val_loss_5.loss_class.average
            )
        print(info)
        print('\033[32m -----Finish calculate mAP-------\033[0m')

        if mAP > self.best_mAP:
            self.best_mAP = mAP
            self.saver_best.save(
                self.sess,
                train_args.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'
                .format(self.epoch, int(__global_step), self.best_mAP, val_loss_5.loss_total.average, __lr)  # todo
            )
        self.writer.add_summary(
            make_summary('evaluation/val_mAP', mAP), global_step=self.epoch
        )
        self.writer.add_summary(
            make_summary('evaluation/val_recall', rec_total.average), global_step=self.epoch
        )
        self.writer.add_summary(
            make_summary('evaluation/val_precision', prec_total.average), global_step=self.epoch
        )
        self.writer.add_summary(
            make_summary('validation_statistics/total_loss', val_loss_5.loss_total.average), global_step=self.epoch
        )
        self.writer.add_summary(
            make_summary('validation_statistics/loss_xy', val_loss_5.loss_xy.average), global_step=self.epoch
        )
        self.writer.add_summary(
            make_summary('validation_statistics/loss_wh', val_loss_5.loss_wh.average), global_step=self.epoch
        )
        self.writer.add_summary(
            make_summary('validation_statistics/loss_conf', val_loss_5.loss_conf.average), global_step=self.epoch
        )
        self.writer.add_summary(
            make_summary('validation_statistics/loss_class', val_loss_5.loss_class.average), global_step=self.epoch
        )
        print('\033[32m -----Finish evaluating in val data-----------\033[0m')


if __name__ == '__main__':
    train = Train()
    train.train()
