# coding: utf-8


from utils.dataset_tricks import create_iterator
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

# setting placeholders
is_training = tf.placeholder(tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
# register the gpu nms operation here for the following evaluation scheme
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
score_threshold = setting.get_score_threshold(is_training=True)
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, setting.class_num, train_setting.nms_topk, score_threshold, setting.nms_threshold)

train_init_op, val_init_op, image_ids, image, y_true = create_iterator()

##################
# Model definition
##################

with tf.variable_scope('yolov3'):
    yolo_model = YoloV3(is_training=True)

yolo_model.compute_loss(y_true)
loss = yolo_model.loss
pred_feature_maps = yolo_model.feature_map_1, yolo_model.feature_map_2, yolo_model.feature_map_3
y_pred = yolo_model.boxes, yolo_model.scores, yolo_model.labels

l2_loss = tf.losses.get_regularization_loss()

# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(
    var_list=tf.contrib.framework.get_variables_to_restore(
        include=train_setting.restore_include, exclude=train_setting.restore_exclude
    )
)
update_vars = tf.contrib.framework.get_variables_to_restore(include=train_setting.update_part)

tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])

global_step = tf.Variable(
    float(train_setting.global_step),
    trainable=False,
    collections=[tf.GraphKeys.LOCAL_VARIABLES]
)

learning_rate = get_learning_rate(global_step)


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


if not train_setting.save_optimizer:
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

optimizer = config_optimizer(train_setting.optimizer_name, learning_rate)

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
    # apply gradient clip to avoid gradient exploding
    gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
    clip_grad_var = [gv if gv[0] is None else [
          tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
    train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

if not train_setting.save_optimizer:
    print('Saving optimizer parameters to checkpoint! Remember to restore the global_step in the fine-tuning afterwards.')
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver_to_restore.restore(sess, setting.weights_path)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(train_setting.log_dir, sess.graph)

    print('\n----------- start to train -----------\n')

    best_mAP = -np.Inf

    for epoch in range(train_setting.total_epoches):

        sess.run(train_init_op)
        loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for i in trange(train_setting.train_batch_num):
            image, __y_true = sess.run(image, y_true)
            _, summary, __y_pred, __loss, __global_step, __lr = sess.run(
                [train_op, merged, y_pred, loss, global_step, learning_rate],
                feed_dict={yolo_model.input_data: image})

            writer.add_summary(summary, global_step=__global_step)

            loss_total.update(__loss[0], len(__y_pred[0]))
            loss_xy.update(__loss[1], len(__y_pred[0]))
            loss_wh.update(__loss[2], len(__y_pred[0]))
            loss_conf.update(__loss[3], len(__y_pred[0]))
            loss_class.update(__loss[4], len(__y_pred[0]))

            if __global_step % train_setting.train_evaluation_step == 0 and __global_step > 0:
                # recall, precision = evaluate_on_cpu(__y_pred, __y_true, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)
                recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred, __y_true, setting.class_num, setting.nms_threshold)

                info = "Epoch: {}, global_step: {} | loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | ".format(
                        epoch, int(__global_step), loss_total.average, loss_xy.average, loss_wh.average, loss_conf.average, loss_class.average)
                info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, __lr)
                print(info)

                writer.add_summary(make_summary('evaluation/train_batch_recall', recall), global_step=__global_step)
                writer.add_summary(make_summary('evaluation/train_batch_precision', precision), global_step=__global_step)

                if np.isnan(loss_total.average):
                    print('****' * 10)
                    raise ArithmeticError(
                        'Gradient exploded! Please train again and you may need modify some parameters.')

        # NOTE: this is just demo. You can set the conditions when to save the weights.
        if epoch % train_setting.save_epoch == 0 and epoch > 0:
            if loss_total.average <= 2.:
                saver_to_save.save(sess, train_setting.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(epoch, int(__global_step), loss_total.average, __lr))

        # switch to validation dataset for evaluation
        if epoch % train_setting.val_evaluation_epoch == 0 and epoch >= train_setting.warm_up_epoch:
            sess.run(val_init_op)

            val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
                AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            val_preds = []

            for j in trange(train_setting.val_img_cnt):
                __image_ids, __y_true, image = sess.run(image_ids, image, y_true)
                __y_pred, __loss = sess.run(
                    [y_pred, loss],
                    feed_dict={yolo_model.input_data: image}
                )
                pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)
                val_preds.extend(pred_content)
                val_loss_total.update(__loss[0])
                val_loss_xy.update(__loss[1])
                val_loss_wh.update(__loss[2])
                val_loss_conf.update(__loss[3])
                val_loss_class.update(__loss[4])

            # calc mAP
            rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
            gt_dict = parse_gt_rec(train_setting.val_file, setting.img_size, setting.letterbox_resize_used)

            info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)

            for ii in range(setting.class_num):
                npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=train_setting.eval_threshold, use_07_metric=train_setting.use_voc_07_metric)
                info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
                rec_total.update(rec, npos)
                prec_total.update(prec, nd)
                ap_total.update(ap, 1)

            mAP = ap_total.average
            info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average, mAP)
            info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'.format(
                val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average)
            print(info)

            if mAP > best_mAP:
                best_mAP = mAP
                saver_best.save(sess, train_setting.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'.format(
                                   epoch, int(__global_step), best_mAP, val_loss_total.average, __lr))

            writer.add_summary(make_summary('evaluation/val_mAP', mAP), global_step=epoch)
            writer.add_summary(make_summary('evaluation/val_recall', rec_total.average), global_step=epoch)
            writer.add_summary(make_summary('evaluation/val_precision', prec_total.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/total_loss', val_loss_total.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss_xy.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss_wh.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss_conf.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_class', val_loss_class.average), global_step=epoch)

