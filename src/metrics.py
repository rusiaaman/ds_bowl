# Define IoU metric
import tensorflow as tf
import keras.backend as K
import numpy as np
def mean_iou2(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def mean_iou(y_true, y_pred):
    y_predict=tf.to_float(y_pred>0.5)
    return K.mean(tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(y_predict,y_true),tf.equal(y_true,1))),axis=(1,2,3))/(1e-30+tf.reduce_sum(tf.maximum(y_predict,y_true),axis=(1,2,3))))

def mean_iou_np(y_true, y_pred):
    y_predict=(y_pred>0.5)
    return np.mean(np.sum(np.logical_and(y_predict==y_true,y_true==1)),axis=(1,2,3))/(1e-30+np.sum(np.maximum(y_predict,y_true),axis=(1,2,3)))
