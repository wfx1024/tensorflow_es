# coding=utf-8
from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import config.setting as setting
import config.train_setting as train_setting
from net.net_module import darknet53, detect_net
from utils.nms import gpu_nms
from utils.eval_utils import get_preds_gpu, evaluate_on_gpu, parse_gt_rec, voc_eval
from utils.dataset_tricks import create_iterator
from utils.training_utils import AverageMeter, get_learning_rate, config_optimizer
from tqdm import trange


slim = tf.contrib.slim


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


class YoloV3:
    """
    Yolo v3
    """
    def __init__(self, is_training=False):
        """
        :param is_training:  是否是训练，控制BN
        """
        self.is_training = is_training

        self.use_label_smooth, self.use_focal_loss, self.use_static_shape, self.batch_norm_decay \
            = setting.get_strategy(is_training)
        # dataset
        if is_training:
            self.train_init_op, self.val_init_op, self.image_ids, self.image, self.y_true\
                = create_iterator()
        self.class_num = setting.class_num  # 类别数量
        self.anchors = setting.anchors  # anchor boxes
        self.weight_decay = setting.weight_decay
        self.freeze_body = False  # 冻结，切换至 Darknet-19
        self.score_threshold = setting.get_score_threshold(is_training)
        self.reuse = setting.reuse
        self.batch_norm_params = {
            'decay': self.batch_norm_decay,  # BN衰减系数
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # tf.nn.fused_batch_norm
        }
        self.best_mAP = - np.Inf  # mAP先设为infinite
        self._build_networks()  # 构建网络
        self._feature_map_to_bboxes()  # 将feature map转换为bboxes信息。
        self.pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
        self.pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        print("Begin initializing variables...")
        self.sess.run(tf.initialize_all_variables())
        print("Finish initializing variables...")
        # if is_training:
            # self._compute_loss()
            # self._optimizer()
        self.writer = tf.summary.FileWriter(train_setting.log_dir, self.sess.graph)
        self.merged = tf.summary.merge_all()

    def _build_networks(self):
        """
        构建DarkNet53
        :return:
        """
        if self.use_static_shape:
            self.input_data = tf.placeholder(tf.float32, [1, setting.img_size[0], setting.img_size[1], 3])
        else:
            self.input_data = tf.placeholder(tf.float32, [None, None, None, 3])
        self.img_size = tf.shape(self.input_data)[1:3]
        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=self.reuse):
            with slim.arg_scope(
                    [slim.conv2d],
                    normalizer_fn=slim.batch_norm,  # Batch Normalize
                    normalizer_params=self.batch_norm_params,
                    biases_initializer=None,
                    activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),  # relu 函数
                    weights_regularizer=slim.l2_regularizer(self.weight_decay)):  # L2正则化
                # DarkNet-53
                with tf.variable_scope('darknet53_body'):
                    # (1, 52, 52, 256)(1, 26, 26, 512)(1, 13, 13, 1024)
                    route_1, route_2, route_3 = darknet53(self.input_data)
                #  构建后续检测模块
                with tf.variable_scope('yolov3_head'):
                    # (1*13*13*255) (1*26*26*255) (1*52*52*255)
                    self.feature_map_1, self.feature_map_2, self.feature_map_3 = detect_net(route_1, route_2, route_3, self.use_static_shape)

    def _feature_map_to_bboxes(self):
        """
        提取的Feature map进行后续步骤，转化为bboxes信息
        :return:
        """
        def _reshape(each_result):
            """
            每种尺度改变形状
            :param each_result:
            :return:
            """
            x_y_offset, boxes, conf_logits, prob_logits = each_result
            grid_size = x_y_offset.get_shape().as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        print("Begin building feature map to bboxes op...")
        feature_map_anchors = [
            (self.feature_map_1, self.anchors[6:9]),  # (116,90), (156,198), (373,326)
            (self.feature_map_2, self.anchors[3:6]),  # (30,61), (62,45), (59,119),
            (self.feature_map_3, self.anchors[0:3])  # (10,13), (16,30), (33,23),
        ]  # 针对不同grid使用不同大小的anchor

        # 对每种尺度进行特征融合
        reorg_results = [self._reorg_layer(feature_map_i, anchors) for (feature_map_i, anchors) in feature_map_anchors]

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
        
        # 三种尺度的结果, 416*416 举例
        # [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2
        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        pred_scores = confs * probs
        self.boxes, self.scores, self.labels = gpu_nms(
            boxes, pred_scores, setting.class_num, max_boxes=setting.max_boxes,
            score_thresh=self.score_threshold, nms_thresh=setting.nms_threshold
        )
        print("Finish building feature map to bboxes op...")

    def predict(self, img_origin):
        """
        预测函数，输入图片，输出bboxes信息
        :param img_origin:
        :return:
        """

        boxes, scores, labels = self.sess.run([self.boxes, self.scores, self.labels], feed_dict={self.input_data: img_origin})
        return boxes, scores, labels

    def _reorg_layer(self, feature_map_i, anchors):
        """
        转移层，特征融合(Fine-Grained Features)，把高分辨率的浅层特征连接到低分辨率的生成特征，堆积在不同channel上。
        :param feature_map_i: 不同尺度的 feature map，
        :param anchors: 3个anchor，shape=3,2
        :return:
        """
        # 选用tf.shape()和tensor.get_shape(),得到grid划分(前者快一点点)
        # 格式为[h, w], 格子划分13*13，,26*26, 52*52
        grid_size = feature_map_i.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map_i)[1:3]
        # 每个cell的大小，转成float32, [w, h]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # 转换anchor数据符合feature map, 注意顺序,
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        # 3个feature map channel都是255.
        # 3*(1+4+4*20)=255, 含义为3个anchor boxes, 4个pre_boxes, 1个置信度confidence, 和80个类别的概率
        # 将feture(1,13,13,255)转为shape=[?, grid_h, grid_w, 3, (5+80)]=(1,13,13,3,85)
        feature_map = tf.reshape(feature_map_i, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])
        # 将其在最后一维分割成4个tensor, 2个[?, grid_h, grid_w, 3, 2], [?, grid_h, grid_w, 3, 1], [?, grid_h, grid_w, 3, 20]
        # 分别是(1,13,13,3,2)(1,13,13,3,2)(1,13,13,3,1)(1,13,13,3,80)
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        # logistic预测的, 所以sigmoid激活
        box_centers = tf.nn.sigmoid(box_centers)

        # 通过一些广播技巧，获得网格的坐标
        grid_x = tf.range(grid_size[1], dtype=tf.int32)  # 0-13
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)  # 两个13*13的tensor,grid_x每行0-13, grid_y每列0-13
        x_offset = tf.reshape(grid_x, (-1, 1))  # 169*1
        y_offset = tf.reshape(grid_y, (-1, 1))  # 169*1
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)  # 169*2
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)  # shape:[13, 13, 1, 2]

        # 获得框在feature map上的绝对坐标, 转换为原图坐标
        box_centers = box_centers + x_y_offset
        box_centers = box_centers * ratio[::-1]

        # tf.clip_by_value避免nan值
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # 转换成初始图片尺度
        box_sizes = box_sizes * ratio[::-1]

        # [N, 13, 13, 3, 4]
        # 最后的维度: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4],  转换成初始图片尺度
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def _box_iou(self, pred_boxes, valid_true_boxes):
        """
        计算交并比
        :param pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
        :param valid_true_boxes: [V, 4]
        :return:
        """

        # [13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # [V, 2]
        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)
        return iou

    def _loss_layer(self, feature_map_i, y_true, anchors):
        """
        计算损失函数
        :param feature_map_i:  feature maps [N, 13, 13, 3*(5 + num_class)]
        :param y_true: y_ture [N, 13, 13, 3, 5 + num_class + 1]
        :param anchors: [3, 2]
        :return:
        """
        grid_size = tf.shape(feature_map_i)[1:3]  # 尺寸[h, w]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)  # 高宽缩放比例
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)  # batch_size
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self._reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########

        # 输入416x416和13*13的feature_map举例
        object_mask = y_true[..., 4:5]  # [N, 13, 13, 3, 1]

        # the calculation of ignore mask if referred from
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))

        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = self._box_iou(pred_boxes[idx], valid_true_boxes)
            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask

        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment: 
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # 损失
        ############
        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        if self.use_label_smooth:  # whether to use label smooth
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def compute_loss(self, y_true):
        """
        计算损失
        :return:
        """
        print("Begin building compute loss op...")
        # self.y_true = tf.placeholder(tf.float32, [None, None, None, None, None])
        feature_map = self.feature_map_1, self.feature_map_2, self.feature_map_3
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # 计算3种维度的5种损失
        for i in range(len(feature_map)):
            result = self._loss_layer(feature_map[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        self.loss = [total_loss, loss_xy, loss_wh, loss_conf, loss_class]
        print("Finish building compute loss op...")

    # def _optimizer(self):
    #     """
    #     构建优化器
    #     :return:
    #     """
    #     print("Begin building optimizer...")
    #     # 步数
    #     self.global_step = tf.Variable(
    #         float(train_setting.global_step),
    #         collections=[tf.GraphKeys.LOCAL_VARIABLES],
    #         trainable=False
    #     )
    #     self.l2_loss = tf.losses.get_regularization_loss()  # L2损失
    #     tf.summary.scalar('train_batch_statistics/total_loss', self.loss[0])
    #     tf.summary.scalar('train_batch_statistics/loss_xy', self.loss[1])
    #     tf.summary.scalar('train_batch_statistics/loss_wh', self.loss[2])
    #     tf.summary.scalar('train_batch_statistics/loss_conf', self.loss[3])
    #     tf.summary.scalar('train_batch_statistics/loss_class', self.loss[4])
    #     tf.summary.scalar('train_batch_statistics/loss_l2', self.l2_loss)
    #
    #     update_vars = tf.contrib.framework.get_variables_to_restore(include=train_setting.update_part)
    #
    #     # 学习率
    #     self.learning_rate = get_learning_rate(self.global_step)
    #
    #     # 优化器
    #     if not train_setting.save_optimizer:
    #         self.saver_to_save = tf.train.Saver()
    #         self.saver_best = tf.train.Saver()
    #     optimizer = config_optimizer(train_setting.optimizer_name, self.learning_rate)
    #
    #     # BN操作
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         # 梯度下降
    #         gvs = optimizer.compute_gradients(self.loss[0] + self.l2_loss, var_list=update_vars)
    #         # 应用gradient clip, 防止梯度爆炸
    #         clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
    #         self.train_op = optimizer.apply_gradients(clip_grad_var, global_step=self.global_step)
    #
    #     if train_setting.save_optimizer:
    #         print('保存optimizer参数到checkpoint! 在fine-tuning后续步骤中 restore global_step')
    #         self.saver_to_save = tf.train.Saver()
    #         self.saver_best = tf.train.Saver()
    #     print("Finish building optimizer...")
    #
    # def train(self):
    #     """
    #     训练
    #     :return:
    #     """
    #     # 保存各种参数
    #     # self.saver
    #     # saver_to_restore = tf.train.Saver(
    #     #     var_list=tf.contrib.framework.get_variables_to_restore(
    #     #         include=train_setting.restore_include, exclude=train_setting.restore_exclude
    #     #     )
    #     # )
    #     # self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #     # saver_to_restore.restore(self.sess, setting.weights_path)  # 加载权重
    #     print('\n----------- start to train -----------\n')
    #     for epoch in range(train_setting.total_epoches):  # 训练100epoch
    #         self.sess.run(self.train_init_op)
    #         # 初始化五种损失函数
    #         self.loss_total, self.loss_xy, self.loss_wh, self.loss_conf, self.loss_class \
    #             = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    #
    #         for _ in trange(train_setting.train_batch_num):  # batch
    #             self.y_pred = self.boxes, self.scores, self.labels
    #             self.image = self.sess.run(self.image)
    #             _, summary, y_pred, y_true, loss, global_step, learning_rate = self.sess.run(
    #                 [self.train_op, self.merged, self.y_pred, self.y_true, self.loss, self.global_step, self.learning_rate],
    #                 feed_dict={self.input_data: self.image}
    #             )
    #             self._evaluate_each_batch(epoch, summary, y_pred, y_true, loss, global_step, learning_rate)
    #
    #         self._save_weight(epoch, global_step, learning_rate)
    #         self._evaluate_in_val(epoch, global_step, learning_rate)
    #
    # def _resize_img(self):
    #     return None
    #
    # def _save_weight(self, epoch, global_step, learning_rate):
    #     """
    #     保存权重
    #     :param epoch: epoch index
    #     :param global_step: 步数
    #     :param learning_rate: 学习率
    #     :return:
    #     """
    #     if epoch % train_setting.save_epoch == 0 and epoch > 0:
    #         if self.loss_total.average <= 2.:
    #             self.saver_to_save.save(
    #                 self.sess,
    #                 train_setting.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'
    #                 .format(epoch, int(global_step), self.loss_total.average, learning_rate)
    #             )
    #
    # def _evaluate_each_batch(self, epoch, summary, y_pred, y_true, loss, global_step, learning_rate):
    #     """
    #     每个batch的评估方法
    #     :param epoch:
    #     :param summary:
    #     :param y_pred:
    #     :param y_true:
    #     :param loss:
    #     :param global_step:
    #     :param learning_rate:
    #     :return:
    #     """
    #     # 注册GPU nms, 方便后续
    #     self.gpu_nms_op = gpu_nms(
    #         self.pred_boxes_flag, self.pred_scores_flag, setting.class_num,
    #         train_setting.nms_topk, self.score_threshold, setting.nms_threshold
    #     )
    #     self.writer.add_summary(summary, global_step=global_step)
    #     # 更新误差
    #     self.loss_total.update(loss[0], len(y_pred[0]))
    #     self.loss_xy.update(loss[1], len(y_pred[0]))
    #     self.loss_wh.update(loss[2], len(y_pred[0]))
    #     self.loss_conf.update(loss[3], len(y_pred[0]))
    #     self.loss_class.update(loss[4], len(y_pred[0]))
    #
    #     if global_step % train_setting.train_evaluation_step == 0 and global_step > 0:
    #         # 召回率,精确率
    #         recall, precision = evaluate_on_gpu(
    #             self.sess, self.gpu_nms_op, self.pred_boxes_flag, self.pred_scores_flag,
    #             y_pred, y_true, setting.class_num, setting.nms_threshold
    #         )
    #         info = "Epoch:{}, global_step: {} | loss: total: {:.2f}, " \
    #             .format(epoch, int(global_step), self.loss_total.average)
    #         info += "xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | " \
    #             .format(self.loss_xy.average, self.loss_wh.average, self.loss_conf.average, self.loss_class.average)
    #         info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, learning_rate)
    #         print(info)
    #
    #         self.writer.add_summary(
    #             make_summary('evaluation/train_batch_recall', recall),
    #             global_step=global_step
    #         )
    #         self.writer.add_summary(
    #             make_summary('evaluation/train_batch_precision', precision),
    #             global_step=global_step
    #         )
    #
    #         if np.isnan(self.loss_total.average):
    #             raise ArithmeticError('****' * 10 + '\n梯度爆炸，修改参数后重新训练')
    #
    # def _evaluate_in_val(self, epoch, __global_step, __lr):
    #     """
    #     验证集评估评估方法
    #     :param epoch:
    #     :param __global_step:
    #     :param __lr:
    #     :return:
    #     """
    #     if epoch % train_setting.val_evaluation_epoch == 0 and epoch >= train_setting.warm_up_epoch:
    #         self.sess.run(self.val_init_op)
    #
    #         # 初始化五种代价函数
    #         val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
    #             AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    #
    #         val_preds = []
    #
    #         for _ in trange(train_setting.val_img_cnt):
    #             __image_ids, __y_pred, __loss = self.sess.run(
    #                 [self.image_ids, self.y_pred, self.loss],
    #                 feed_dict={self.input_data: self.image}
    #             )
    #             pred_content = get_preds_gpu(
    #                 self.sess, self.gpu_nms_op, self.pred_scores_flag, __image_ids, __y_pred
    #             )
    #             val_preds.extend(pred_content)
    #             val_loss_total.update(__loss[0])
    #             val_loss_xy.update(__loss[1])
    #             val_loss_wh.update(__loss[2])
    #             val_loss_conf.update(__loss[3])
    #             val_loss_class.update(__loss[4])
    #
    #         # 计算 mAP
    #         rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    #         gt_dict = parse_gt_rec(train_setting.val_file, setting.img_size, setting.letterbox_resize_used)
    #
    #         info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)
    #
    #         for ii in range(setting.class_num):
    #             npos, nd, rec, prec, ap = voc_eval(
    #                 gt_dict, val_preds, ii, iou_thres=train_setting.eval_threshold,
    #                 use_07_metric=train_setting.use_voc_07_metric
    #             )
    #             info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
    #             rec_total.update(rec, npos)
    #             prec_total.update(prec, nd)
    #             ap_total.update(ap, 1)
    #
    #         mAP = ap_total.average
    #         info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'\
    #             .format(rec_total.average, prec_total.average, mAP)
    #
    #         info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n' \
    #             .format(val_loss_total.average, val_loss_xy.average, val_loss_wh.average,
    #                     val_loss_conf.average, val_loss_class.average)
    #
    #         print(info)
    #
    #         if mAP > self.best_mAP:
    #             self.best_mAP = mAP
    #             self.saver_best.save(
    #                 self.sess,
    #                 train_setting.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'
    #                 .format(epoch, int(__global_step), self.best_mAP, val_loss_total.average, __lr)
    #             )
    #         self.writer.add_summary(
    #             make_summary('evaluation/val_mAP', mAP),
    #             global_step=epoch
    #         )
    #         self.writer.add_summary(
    #             make_summary('evaluation/val_recall', rec_total.average),
    #             global_step=epoch
    #         )
    #         self.writer.add_summary(make_summary(
    #             'evaluation/val_precision', prec_total.average),
    #             global_step=epoch
    #         )
    #         self.writer.add_summary(
    #             make_summary('validation_statistics/total_loss', val_loss_total.average),
    #             global_step=epoch
    #         )
    #         self.writer.add_summary(
    #             make_summary('validation_statistics/loss_xy', val_loss_xy.average),
    #             global_step=epoch
    #         )
    #         self.writer.add_summary(
    #             make_summary('validation_statistics/loss_wh', val_loss_wh.average),
    #             global_step=epoch
    #         )
    #         self.writer.add_summary(
    #             make_summary('validation_statistics/loss_conf', val_loss_conf.average),
    #             global_step=epoch)
    #         self.writer.add_summary(
    #             make_summary('validation_statistics/loss_class', val_loss_class.average),
    #             global_step=epoch)
    #
