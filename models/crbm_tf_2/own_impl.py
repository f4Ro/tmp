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
        h_bias_initial_val = tf.keras.initializers.Constant(0)(shape=(1,10,1,24))
        v_bias_initial_val = tf.keras.initializers.Constant(0)(shape=(1,120,1,1))
        self.kernel = tf.Variable(name='weights', initial_value=kernel_initial_val, dtype=tf.dtypes.float32)
        self.h_bias = tf.Variable(name='h_bias', initial_value=h_bias_initial_val, dtype=tf.dtypes.float32)
        self.v_bias = tf.Variable(name='v_bias', initial_value=v_bias_initial_val, dtype=tf.dtypes.float32)

    # =======================================================================================================
    # Gibb's
    # =======================================================================================================
    def gibbs_sampling(self: CRBM, input, n=1):
        V0 = input
        Q0 = self._propagate_forward(input)
        ret = self.sample_forward(Q0)
        for _ in range(n):
            VN = self.sample_backward((ret))
            QN = self._propagate_backward(VN)
            ret = self.sample_backward(QN)
        return V0, Q0, VN, QN

    # =======================================================================================================
    # Sampling logic
    # =======================================================================================================
    def _propagate_forward(self: CRBM, input):
        foo = tf.nn.conv2d(input, self.kernel, [12, 1], padding='SAME')
        return tf.sigmoid(foo + self.h_bias)

    def sample_forward(self: CRBM, hidden_activations):
        hidden_shape = (1, 10, 1, 24) # TODO: change this later to [self.batch_size, height, width, channels]
        samples_rand_uniform = tf.random.uniform(hidden_shape)
        choices_for_true = tf.ones(hidden_shape)
        choices_for_false = tf.zeros(hidden_shape)
        return tf.where(samples_rand_uniform - hidden_activations < 0, choices_for_true, choices_for_false) # TODO make this non-binary later

    def _propagate_backward(self: CRBM, hidden):
        foo = tf.nn.conv2d_transpose(hidden, self.kernel, (1, 120, 1, 1), [1, 12, 1, 1])
        return tf.sigmoid(foo + self.v_bias)

    def sample_backward(self: CRBM, visible_activations):
        # TODO: might want to add sampling here. Usually people do not sample on backward but author did
        return visible_activations

# ===========================================================================================================
# Evaluation
# ===========================================================================================================
batch_size = 1
sequence_length = 120

inputs = tf.random.uniform(
    (batch_size, sequence_length, 1, 1), minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)

c = CRBM()
res = c.gibbs_sampling(inputs)
for r in res:
    print(r.shape)
