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
    def __init__(self: CRBM, batch_size, sequence_length, kernel_height, vertical_stride, v_bias_initializer):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.kernel_height = kernel_height
        self.vert_stride = vertical_stride

        self.hidden_height = int((sequence_length - kernel_height) / vertical_stride + 1)

        self.gaussian_unit = True
        self.padding = False
        self.gaussian_variance = 0.2
        self.sparsity_coef = 0.1
        self.sparsity_target = 0.1
        self.weight_decay = 0.1
        self.learning_rate = 0.0001
        self.momentum_coefficient = 0.9
        self.sigma = tf.Variable(name='sigma', initial_value=tf.constant_initializer(self.gaussian_variance * self.gaussian_variance)(1*120*self.batch_size*1), dtype = tf.float32)

        vitesse_initializer = tf.keras.initializers.Constant(0)
        # vitesse_v_bias_initializer = tf.keras.initializers.Constant(v_bias_initializer)
        self.vitesse_kernels = tf.Variable(name='vitesse_weights', initial_value=vitesse_initializer(shape=(self.kernel_height, 1, 1, 24)))
        self.vitesse_h_bias = tf.Variable(name='vitesse_h_bias', initial_value=vitesse_initializer(shape=(self.batch_size, self.hidden_height, 1, 24)))
        self.vitesse_v_bias = tf.Variable(name='vitesse_v_bias', initial_value=vitesse_initializer(shape=(self.batch_size, self.sequence_length, 1, 1)))

        # height, width, in_channels, out_channels/filter_num
        kernel_initial_val = tf.keras.initializers.truncated_normal(stddev=0.01)(shape=(self.kernel_height, 1, 1, 24))
        h_bias_initial_val = tf.keras.initializers.Constant(0)(shape=(self.batch_size, self.hidden_height, 1, 24))
        v_bias_initial_val = tf.keras.initializers.Constant(v_bias_initializer)(shape=(self.batch_size, self.sequence_length, 1, 1))
        self.kernel = tf.Variable(name='weights', initial_value=kernel_initial_val, dtype=tf.dtypes.float32)
        self.h_bias = tf.Variable(name='h_bias', initial_value=h_bias_initial_val, dtype=tf.dtypes.float32)
        self.v_bias = tf.Variable(name='v_bias', initial_value=v_bias_initial_val, dtype=tf.dtypes.float32)

    # =======================================================================================================
    # Contrastive Divergence
    # =======================================================================================================
    def contrastive_divergence(self: CRBM, input):
        V0, Q0, VN, QN = self.gibbs_sampling(input)

        #                                                 [height   ,width      , in_chann ,out_chann]            []
        # [vchannels,vheight,vwidth,batch_size] conv with [hiddenhei,hiddenwidth,batch_size,hiddenchannel] give a [vchannels,filterhei,filterwidth,hiddenchannel] filter
        reshaped_V0 = tf.transpose(V0, perm=[3, 1, 2, 0])
        reshaped_Q0 = tf.transpose(Q0, perm=[1, 2, 0, 3])
        reshaped_VN = tf.transpose(VN, perm=[3, 1, 2, 0])
        reshaped_QN = tf.transpose(QN, perm=[1, 2, 0, 3])
        # print(reshaped_V.shape)
        # print(reshaped_H.shape)
        positive = tf.nn.conv2d(self._pad(reshaped_V0), reshaped_Q0, [1, self.vert_stride, 1, 1], padding='VALID')
        negative = tf.nn.conv2d(self._pad(reshaped_VN), reshaped_QN, [1, self.vert_stride, 1, 1], padding='VALID')
        ret = positive - negative

        ###
        if self.gaussian_unit:
            ret = tf.divide(ret, self.gaussian_variance)
        g_weight = tf.divide(tf.transpose(ret, perm=[1, 2, 0, 3]), self.batch_size)
        g_weight_sparsity = self._get_param_sparsity_penalty('kernel', Q0, V0)
        g_weight_l2 = self._get_param_weight_penalty(self.kernel)

        # 'FOR BIAS VISIBLE'
        g_biais_V = tf.divide(tf.reduce_sum(tf.subtract(V0, VN), [0, 1, 2]), self.batch_size)
        if self.gaussian_unit:
            g_biais_V = tf.divide(g_biais_V, self.gaussian_variance * self.gaussian_variance)

        # 'FOR BIAS HIDDEN'
        g_biais_H = tf.divide(tf.reduce_sum(tf.subtract(Q0, QN), [0, 1, 2]), self.batch_size)
        g_biais_H_sparsity = self._get_param_sparsity_penalty('hidden_bias', Q0, V0)

        # 'UPDATE ALL'
        ret_w = self._apply_grad(
            self.kernel, g_weight, self.vitesse_kernels, use_weight_decay=False, weight_decay_value=g_weight_l2, sparsity=False,
            sparsity_value=g_weight_sparsity, global_step=0)
        ret_bv = self._apply_grad(self.v_bias, g_biais_V, self.vitesse_v_bias,
                                  global_step=0)
        ret_bh = self._apply_grad(self.h_bias, g_biais_H, self.vitesse_h_bias, sparsity=True,
                                  sparsity_value=g_biais_H_sparsity, global_step=0)
        cost = tf.reduce_sum(tf.square(tf.subtract(input, VN)))
        update = tf.reduce_sum(VN)
        return ret_w, ret_bv, ret_bh, cost, update

    # =======================================================================================================
    # Gibb's
    # =======================================================================================================
    def gibbs_sampling(self: CRBM, input, n=1):
        V0 = input
        Q0 = self._propagate_forward(input)
        ret = self.sample_forward(Q0)
        for _ in range(n):
            VN = self.sample_backward(self._propagate_backward(ret))
            QN = self._propagate_forward(VN)
            ret = self.sample_forward(QN)
        return V0, Q0, VN, QN

    # =======================================================================================================
    # Sampling logic
    # =======================================================================================================
    def _propagate_forward(self: CRBM, input):
        foo = tf.nn.conv2d(input, self.kernel, [self.vert_stride, 1], padding='VALID')
        return tf.sigmoid(foo + self.h_bias)

    def sample_forward(self: CRBM, hidden_activations):
        hidden_shape = (self.batch_size, self.hidden_height, 1, 24) # TODO: change this later to [self.batch_size, height, width, channels]
        samples_rand_uniform = tf.random.uniform(hidden_shape)
        choices_for_true = tf.ones(hidden_shape)
        choices_for_false = tf.zeros(hidden_shape)
        return tf.where(samples_rand_uniform - hidden_activations < 0, choices_for_true, choices_for_false) # TODO make this non-binary later

    def _propagate_backward(self: CRBM, hidden):
        foo = tf.nn.conv2d_transpose(hidden, self.kernel, (self.batch_size, self.sequence_length, 1, 1), [1, self.vert_stride, 1, 1], padding='VALID')
        return tf.sigmoid(foo + self.v_bias)

    def sample_backward(self: CRBM, visible_activations):
        # mu = tf.reshape(visible_activations, [-1])
        # dist = tfp.distributions.MultivariateNormalDiag(mu, self.sigma)
        # samples = dist.sample()
        # return tf.reshape(samples,[self.batch_size, 120, 1, 1])
        # TODO: might want to add sampling here. Usually people do not sample on backward but author did
        return visible_activations

    # =======================================================================================================
    # Penalties
    # =======================================================================================================
    def _get_param_sparsity_penalty(self: CRBM, name, Q0, V0):
        """INTENT : Compute sparsity penalty term (l2 norm of target minus mean activation)
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        name               :        parameter that we want to compute it sparsity pernalty term (hidden layer bias   or kenel weight)
        Q0                 :        infered probabilities for hidden layer
        V0                 :        visible layer value (input)
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : see http://ai.stanford.edu/~ang/papers/nips07-sparsedeepbeliefnetworkv2.pdf
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : formula is for hidden_bias: -2lambda*sumonwidth/height[meanonbatch[(1-Q0)Q0(target-meanonbatch[Q0])]]
                            for kernel      -2lambda*sumonwidth/height[meanonbatch[(1-Q0)Q0(target-meanonbatch[Q0])v]]     vi+r-1,j+s-1,(l) = v"""

        ret = -2 * self.sparsity_coef / self.batch_size
        if self.gaussian_unit:
            ret = ret / self.gaussian_variance
        mean = tf.reduce_sum(Q0, [0], keepdims=True)
        baseline = tf.multiply(tf.subtract(self.sparsity_target, mean), tf.multiply(tf.subtract(1.0, Q0), Q0))
        if name == 'hidden_bias':
            return tf.multiply(ret, tf.reduce_sum(baseline, [0, 1, 2]))
        if name == 'kernel':
            if self.padding:
                retBis = tf.nn.conv2d(
                    self._get_padded_visible(tf.transpose(V0, perm=[3, 1, 2, 0])),
                    tf.transpose(baseline, perm=[1, 2, 0, 3]),
                    [1, self.vert_stride, 1, 1],
                    padding='VALID')
            else:
                retBis = tf.nn.conv2d(
                    tf.transpose(self._pad(V0), perm=[3, 1, 2, 0]),
                    tf.transpose(baseline, perm=[1, 2, 0, 3]),
                    [1, self.vert_stride, 1, 1],
                    padding='VALID')
            retBis = tf.transpose(retBis, perm=[1, 2, 0, 3])
            return tf.multiply(ret, retBis)

    def _get_param_weight_penalty(self: CRBM, operand):
        """INTENT : Compute l2 regularization pernalty term
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        operand               :        parameter to be regularized"""

        return tf.multiply(self.weight_decay, operand)

    def _apply_grad(
            self: CRBM, parameter, grad_value, vitesse, use_weight_decay=False, weight_decay_value=None, sparsity=False, sparsity_value=None,
            global_step=0):
        """INTENT : Apply gradient descent to provided variable. Also modify it vitesse
        ------------------------------------------------------------------------------------------------------------------------------------------
        """

        # lr = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_step,
        #                                 self.learning_rate_decay, staircase=True)
        lr = self.learning_rate
        momentum_term = tf.multiply(self.momentum_coefficient, vitesse)
        update_term =tf.multiply(lr, grad_value)
        ret = tf.add(momentum_term, update_term)
        # print(ret[:, : ,:, :2])
        ret = vitesse.assign(ret)
        if use_weight_decay:
            ret = tf.subtract(ret, tf.multiply(lr, weight_decay_value))
        if sparsity:
            ret = tf.subtract(ret, tf.multiply(lr, sparsity_value))
        return parameter.assign_add(ret)

    # =======================================================================================================
    # Sampling logic
    # =======================================================================================================
    def _pad(self: CRBM, input):
        return input
        return tf.pad(
            input,
            [
                [0, 0],
                [11, 11],
                [0, 0],
                [0, 0]
            ]
        )

# ===========================================================================================================
# Evaluation
# ===========================================================================================================
batch_size = 1
sequence_length = 120
from data_preprocessing.berkley_lab_data import read_and_preprocess_data
x_train, x_test = read_and_preprocess_data(
    should_smooth=False,
    smoothing_window=100,
    sequence_length=sequence_length,
    cut_off_min=5,
    cut_off_max=45,
    should_scale=True,
    data_path="datasets/data.txt",
    batch_size=batch_size,
    motes_train=[7],
    motes_test=[7]
)

inp = x_train[:1, :, :]
mean = np.mean(inp)
inp = tf.cast(tf.reshape(tf.constant(inp), (1, 120, 1, 1)), tf.float32)
# inp = tf.ones((1, 120, 1, 1))
# mean = 1

c = CRBM(
    batch_size=batch_size, sequence_length=sequence_length,
    kernel_height=12, vertical_stride=12,
    v_bias_initializer=mean
)

def plot_weights(dir: str):
    weights = c.kernel

    _, ax = plt.subplots(24, figsize=(15,30))
    for x in range(24):
        curr_weight = weights.numpy()[:, :, :, x].reshape(-1)
        ax[x].plot(curr_weight)
    harry('weights', f'/{dir}')

# Quickly hacked together, overfitting the C-RBM on a single example
# Checkin performance before
def predict_and_plot(input, name):
    p1 = c._propagate_forward(input)
    a1 = c.sample_forward(p1)

    pv1 = c._propagate_backward(a1)
    av1 = c.sample_backward(pv1)

    plt.plot(input.numpy().reshape(-1), label='original')
    plt.plot(av1.numpy().reshape(-1), label='reconstruction')
    plt.legend()
    harry(name)
predict_and_plot(inp, 'before')

old_weights, old_h_bias, old_v_bias = tf.identity(c.kernel), tf.identity(c.h_bias), tf.identity(c.v_bias)
# Training for some epochs
for y in range(200):
    # predict_and_plot(inp, str(y))
    # plot_weights(y)
    print(c.contrastive_divergence(inp)[3].numpy())

new_weights, new_h_bias, new_v_bias = tf.identity(c.kernel), tf.identity(c.h_bias), tf.identity(c.v_bias)

_, ax = plt.subplots(24, figsize=(15,30))
for x in range(24):
    old = old_weights.numpy()[:, :, :, x].reshape(-1)
    new = new_weights.numpy()[:, :, :, x].reshape(-1)
    ax[x].plot(old, label='old')
    ax[x].plot(new, label='new')
    ax[x].legend()
harry('weights')
# Checking performance afterwards
predict_and_plot(inp, 'after')
