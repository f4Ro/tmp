from __future__ import annotations, division

# Set seeds
from numpy.random import seed
from tensorflow.random import set_seed
seed(1)
set_seed(10)

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from util.plotter import Plotter
from shared_code.per_rms_diff import per_rms_diff

harry = Plotter('CRBM_dev', plt)

class CRBM:
    def __init__(self: CRBM):
        # height, width, in_channels, out_channels/filter_num
        kernel_initial_val = tf.keras.initializers.truncated_normal(stddev=0.01)(shape=(12, 1, 1, 24))
        self.kernel = tf.Variable(name='weights', initial_value=kernel_initial_val, dtype=tf.dtypes.float32)

    def _propagate_forward(self: CRBM, input):
        return tf.nn.conv2d(input, self.kernel, [12, 1], padding='SAME')

    def _propagate_backward(self: CRBM, hidden):
        return tf.nn.conv2d_transpose(hidden, self.kernel, (1, 120, 1, 1), [1, 12, 1, 1])

# ===========================================================================================================
# Evaluation
# ===========================================================================================================
batch_size = 1
sequence_length = 120

inputs = tf.random.uniform(
    (batch_size, sequence_length, 1, 1), minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)

c = CRBM()
hidden = c._propagate_forward(inputs); print(hidden.shape)
visible = c._propagate_backward(hidden); print(visible.shape)
