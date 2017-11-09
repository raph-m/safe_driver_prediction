import tensorflow as tf


def my_loss(actual, pred):
    return tf.reduce_mean(-tf.log(pred) * actual * 19 - tf.log(1-pred) * (1 - actual))

