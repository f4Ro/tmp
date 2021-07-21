import numpy as np
import tensorflow as tf
from typing import Any

def per_rms_diff(label: Any, prediction: Any):
    diff = np.sum(np.square(np.subtract(label, prediction)))
    sq = np.sum(np.square(label))
    return 100 * (np.sqrt(np.divide(diff, sq + 0.e-6)))



def tf_per_rms_diff(label, pred):
    diff = tf.reduce_sum(tf.square(tf.subtract(label, pred)))
    sq = tf.reduce_sum(tf.square(label))
    return 100 * (tf.sqrt(tf.divide(diff, sq + 0.e-6)))
