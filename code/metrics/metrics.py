from __future__ import division
import numpy as np
from keras import backend as K
dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    import theano
    from theano import tensor as T
else:
    import tensorflow as tf
    from tensorflow.python.framework import ops

def jaccard_coef(y_true, y_pred, eps=1e-37):
    y_pred = tf.one_hot(tf.argmax(y_pred,axis=2), tf.shape(y_pred)[2])
    intersection = K.sum(y_true * y_pred, axis=(1))
    union = K.sum(y_true, axis=(1)) + K.sum(y_pred, axis=(1)) - intersection
    mean  = K.mean( (intersection ) / (union + eps), axis=(1))
    mean  = tf.clip_by_value(mean, 0.0, 1.0)
    return K.mean( mean, axis=0)

def jaccard_coef_smooth(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    intersection = K.sum(y_true * y_pred, axis=(1))
    union = K.sum(y_true, axis=(1)) + K.sum(y_pred, axis=(1)) - intersection
    mean  = K.mean( (intersection + smooth) / (union + smooth), axis=(1))
    return K.mean( mean, axis=0)

def cce_flatt(void_class, weights_class):
    def categorical_crossentropy_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)

        if dim_ordering == 'th':
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        else:
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01

        # remove void classes from cross_entropy
        if len(void_class):
            for i in range(len(void_class)):
                # get idx of non void classes and remove void classes
                # from y_true and y_pred
                idxs = K.not_equal(y_true, void_class[i])
                if dim_ordering == 'th':
                    idxs = idxs.nonzero()
                    y_pred = y_pred[idxs]
                    y_true = y_true[idxs]
                else:
                    y_pred = tf.boolean_mask(y_pred, idxs)
                    y_true = tf.boolean_mask(y_true, idxs)

        if dim_ordering == 'th':
            y_true = T.extra_ops.to_one_hot(y_true, nb_class=y_pred.shape[-1])
        else:
            y_true = tf.one_hot(y_true, K.shape(y_pred)[-1], on_value=1, off_value=0, axis=None, dtype=None, name=None)
            y_true = K.cast(y_true, 'float32')  # b,01 -> b01
        out = K.categorical_crossentropy(y_pred, y_true)

        # Class balancing
        if weights_class is not None:
            weights_class_var = K.variable(value=weights_class)
            class_balance_w = weights_class_var[y_true].astype(K.floatx())
            out = out * class_balance_w

        return K.mean(out)  # b01 -> b,01
    return categorical_crossentropy_flatt


def IoU(n_classes, void_labels):
    def IoU_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)
        y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        y_pred = K.argmax(y_pred, axis=-1)

        # We use not_void in case the prediction falls in the void class of
        # the groundtruth
        for i in range(len(void_labels)):
            if i == 0:
                not_void = K.not_equal(y_true, void_labels[i])
            else:
                not_void = not_void * K.not_equal(y_true, void_labels[i])

        sum_I = K.zeros((1,), dtype='float32')

        out = {}
        for i in range(n_classes):
            y_true_i = K.equal(y_true, i)
            y_pred_i = K.equal(y_pred, i)

            if dim_ordering == 'th':
                I_i = K.sum(y_true_i * y_pred_i)
                U_i = K.sum(T.or_(y_true_i, y_pred_i) * not_void)
                # I = T.set_subtensor(I[i], I_i)
                # U = T.set_subtensor(U[i], U_i)
                sum_I = sum_I + I_i
            else:
                U_i = K.sum(K.cast(tf.logical_and(tf.logical_or(y_true_i, y_pred_i), not_void), 'float32'))
                y_true_i = K.cast(y_true_i, 'float32')
                y_pred_i = K.cast(y_pred_i, 'float32')
                I_i = K.sum(y_true_i * y_pred_i)
                sum_I = sum_I + I_i
            out['I'+str(i)] = I_i
            out['U'+str(i)] = U_i

        if dim_ordering == 'th':
            accuracy = K.sum(sum_I) / K.sum(not_void)
        else:
            accuracy = K.sum(sum_I) / tf.reduce_sum(tf.cast(not_void, 'float32'))
        out['acc'] = accuracy
        return out
    return IoU_flatt



"""
    YOLO loss function
    code adapted from https://github.com/thtrieu/darkflow/
"""

def YOLOLoss(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,object_scale=5.0,noobject_scale=1.0,coord_scale=1.0,class_scale=1.0):

  # Def custom loss function using numpy
  def _YOLOLoss(y_true, y_pred, name=None, priors=priors):

      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU, set 0.0 confidence for worse boxes
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
      #best_box = tf.grater_than(iou, 0.5) # LLUIS ???
      best_box = tf.to_float(best_box)
      confs = tf.multiply(best_box, _confs)

      # take care of the weight terms
      conid = noobject_scale * (1. - confs) + object_scale * confs
      weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
      cooid = coord_scale * weight_coo
      weight_pro = tf.concat(num_classes * [tf.expand_dims(confs, -1)], 3)
      proid = class_scale * weight_pro

      true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
      wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

      loss = tf.square(adjusted_net_out - true)
      loss = tf.multiply(loss, wght)
      loss = tf.reshape(loss, [-1, h*w*b*(4 + 1 + num_classes)])
      loss = tf.reduce_sum(loss, 1)

      return .5*tf.reduce_mean(loss)

  return _YOLOLoss


"""
    YOLO detection metrics
    code adapted from https://github.com/thtrieu/darkflow/
"""
def YOLOMetrics(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3,name='avg_iou'):

  def avg_recall(y_true, y_pred, name=None):
      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      recall        = tf.reduce_sum(tf.to_float(tf.greater(best_ious,0.5)), [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_recall    = tf.truediv(recall, num_gt_obj)
 
      return tf.reduce_mean(avg_recall)

  def avg_iou(y_true, y_pred, name=None):
      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      sum_best_ious = tf.reduce_sum(best_ious, [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_iou       = tf.truediv(sum_best_ious, num_gt_obj)
 
      return tf.reduce_mean(avg_iou)

  if name=='avg_iou':
    return avg_iou
  else:
    return avg_recall


"""
    Multibox loss function for SSD300

The Keras-compatible loss function for the SSD model. Currently supports TensorFlow only.

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

class SSDLoss(object):
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

        # 1: Compute the losses for class and box predictions for every box

        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # Output shape: (batch_size, n_boxes)

        # 2: Compute the classification losses for the positive and negative targets

        # Create masks for the positive and negative ground truth classes
        negatives = y_true[:,:,0] # Tensor of shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1)) # Tensor of shape (batch_size, n_boxes)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes)
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # Compute the classification loss for the negative default boxes (if there are any)

        # First, compute the classification loss for all negative boxes
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False) # We don't need sorting
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets
        #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: Compute the total loss

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        total_loss *= tf.to_float(batch_size)

        return total_loss

"""
    YOLO detection metrics
    code adapted from https://github.com/thtrieu/darkflow/
"""
def SSDMetrics(input_shape=(3,300,300),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3,name='avg_iou'):

  def avg_recall(y_true, y_pred, name=None):
      print "y_true: ", y_true
      print "y_pred: ", y_pred
      
      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      recall        = tf.reduce_sum(tf.to_float(tf.greater(best_ious,0.5)), [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_recall    = tf.truediv(recall, num_gt_obj)
 
      return tf.reduce_mean(avg_recall)

  def avg_iou(y_true, y_pred, name=None):
      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      sum_best_ious = tf.reduce_sum(best_ious, [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_iou       = tf.truediv(sum_best_ious, num_gt_obj)
 
      return tf.reduce_mean(avg_iou)

  if name=='avg_iou':
    return avg_iou
  else:
    return avg_recall