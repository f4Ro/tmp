from __future__ import annotations, division

# Set seeds
from numpy.random import seed; seed(1)
from tensorflow.random import set_seed; set_seed(1)

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from util.plotter import Plotter
from shared_code.per_rms_diff import per_rms_diff
from data_preprocessing.berkley_lab_data import read_and_preprocess_data

harry = Plotter('CRBM_dev', plt)


class CRBM(object):
    """CONVOLUTIONAL RESTRICTED BOLTZMANN MACHINE"""

    def __init__(
            self: CRBM, name: str, fully_connected: bool = False, v_height: int = 1, v_width: int = 1, v_channels: int = 784,
            f_height: int = 1, f_width: int = 1, up_stride: int = 1, side_stride: int = 1, f_number: int = 400, init_biases_H: int = -3, init_biases_V: float = 0.01,
            init_weight_stddev: float = 0.01, gaussian_unit: bool = True, gaussian_variance: float = 0.2,
            prob_maxpooling: bool = False, padding: bool = False, batch_size: int = 20, learning_rate: float = 0.00025, learning_rate_w: float = 0.001,
            learning_rate_decay: float = 0.5, momentum: float = 0.9, decay_step: int = 50000, weight_decay: float = 0.1,
            sparsity_target: float = 0.1, sparsity_coef: float = 0.1) -> None:
        """INTENT : Initialization of a Convolutional Restricted Boltzmann Machine
        ------------------------------------------------------------------
        REMARK : This class can represent differents kinds of RBM
                 fully connected RBM              if  fully_connected = True and then v_height=v_width=f_height=f_width=1
                 gaussian RBM                     if  gaussian_unit = True     (https://web.eecs.umich.edu/~honglak/iccv2011-sparseConvLearning.pdf)
                 sparse   RBM                     if  sparsity_coef <> 0        (http://ai.stanford.edu/~ang/papers/nips07-sparsedeepbeliefnetworkv2.pdf   -    https://www.ii.pwr.edu.pl/~gonczarek/papers/rbm_sparse.pdf)
                 probabilistic maxpooling CRBM    if  prob_maxpooling = True   (http://www.cs.toronto.edu/~rgrosse/icml09-cdbn.pdf)"""

        try:
            if fully_connected and not (v_height == 1 and v_width == 1 and f_height == 1 and f_width == 1):
                raise ValueError(
                    'Trying to initialize CRBM ' + name +
                    ' which is fully connected but height and width of visible and filters are not set to 1')
            if fully_connected and prob_maxpooling:
                raise ValueError(
                    'Trying to initialize CRBM ' + name +
                    ' which is fully connected but with max pooling enabled (should set prob_maxpooling to False)')
            if fully_connected and padding:
                raise ValueError(
                    'Trying to initialize CRBM ' + name +
                    ' which is fully connected but with padding enabled (should set padding to False)')
            if padding and ((f_height % 2 == 0) or (f_width % 2 == 0)):
                raise ValueError(
                    'Trying to initialize CRBM ' + name +
                    ' which has padded enable but filter dimension are not odd (padding feature only support odd size for filter dimension)')

            self.name = name
            self.fully_connected = fully_connected
            self.visible_height = v_height
            self.visible_width = v_width
            self.visible_channels = v_channels
            self.filter_height = f_height
            self.filter_width = f_width
            self.filter_number = f_number
            self.gaussian_unit = gaussian_unit
            if gaussian_unit:
                self.gaussian_variance = gaussian_variance
            self.prob_maxpooling = prob_maxpooling
            self.padding = padding
            if padding:
                self.hidden_height = v_height
                self.hidden_width = v_width
            else:
                # self.hidden_height       = v_height - f_height + 1
                # self.hidden_width        = v_width - f_width + 1
                self.up_stride = up_stride
                self.hidden_height = int((v_height - f_height) / self.up_stride + 1)
                self.hidden_width = 1

            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.learning_rate_w = learning_rate_w
            self.learning_rate_decay = learning_rate_decay
            self.momentum = momentum
            self.decay_step = decay_step
            self.weight_decay = weight_decay
            self.sparsity_target = sparsity_target
            self.sparsity_coef = sparsity_coef

            with tf.device('/cpu:0'):
                self.kernels = tf.Variable(name='weights', initial_value=tf.keras.initializers.truncated_normal(
                    stddev=init_weight_stddev)(shape=(f_height, f_width, v_channels, f_number)))
                # self.kernels = tf.Variable(name='weights', initial_value=tf.keras.initializers.zeros()(shape=(f_height, f_width, v_channels, f_number)))
                self.biases_V = tf.Variable(name='biases_V',
                                            initial_value=tf.keras.initializers.Constant(init_biases_V)
                                            (shape=(v_channels)))
                self.biases_H = tf.Variable(name='biases_H',
                                            initial_value=tf.keras.initializers.Constant(init_biases_H)
                                            (shape=(f_number)))
                self.vitesse_kernels = tf.Variable(name='vitesse_weights', initial_value=tf.keras.initializers.Constant(
                    0)(shape=(f_height, f_width, v_channels, f_number)))
                self.vitesse_biases_V = tf.Variable(name='vitesse_biases_V',
                                                    initial_value=tf.keras.initializers.Constant(0)
                                                    (shape=(v_channels)))
                self.vitesse_biases_H = tf.Variable(name='vitesse_biases_H',
                                                    initial_value=tf.keras.initializers.Constant(0)
                                                    (shape=(f_number)))
                if gaussian_unit:
                    self.sigma = tf.Variable(name='sigma', initial_value=tf.keras.initializers.Constant(
                        self.gaussian_variance * self.gaussian_variance)(shape=(v_width * v_height * batch_size * v_channels)))

        except ValueError as error:
            print('--------------------------')
            print(error.args)
            print('--------------------------')
            raise error

    def infer_probability(self: CRBM, operand, method, result='hidden'):
        """INTENT : Compute the probabily of activation of one layer given the other
        REMARK : If method is FORWARD then compute hidden layer probability and operand is VISIBLE layer
                 If method is BACKWARD then compute visible layer probability and operand is HIDDEN layer"""

        'Computing HIDDEN layer with VISIBLE layer given'
        'Gaussian visible or not, hidden layer activation is a sigmoid'
        if method == 'forward':
            if self.padding:
                conv = tf.nn.conv2d(operand, self.kernels, [1, self.up_stride, 1, 1], padding='SAME')
            else:
                conv = tf.nn.conv2d(operand, self.kernels, [1, self.up_stride, 1, 1], padding='VALID')
            if self.gaussian_unit:
                conv = tf.divide(conv, self.gaussian_variance)
            bias = tf.nn.bias_add(conv, self.biases_H)
            if self.prob_maxpooling:
                raise Exception('foo')
                'SPECIFIC CASE where we enable probabilistic max pooling'
                exp = tf.exp(bias)
                custom_kernel = tf.constant(1.0, shape=[2, 2, self.filter_number, 1])
                sum = tf.nn.depthwise_conv2d(exp, custom_kernel, [1, 2, 2, 1], padding='VALID')
                sum = tf.add(1.0, sum)
                ret_kernel = np.zeros((2, 2, self.filter_number, self.filter_number))
                for i in range(2):
                    for j in range(2):
                        for k in range(self.filter_number):
                            ret_kernel[i, j, k, k] = 1
                custom_kernel_bis = tf.constant(ret_kernel, dtype=tf.float32)
                raise Excpetion('bar')
                sum_bis = tf.nn.conv2d_transpose(
                    sum, custom_kernel_bis,
                    (self.batch_size, self.hidden_height, self.hidden_width, self.filter_number), #self.visible_channels
                    strides=[1, 2, 2, 1],
                    padding='VALID', name=None)
                if result == 'hidden':
                    'We want to obtain HIDDEN layer configuration'
                    return tf.divide(exp, sum_bis)
                elif result == 'pooling':
                    'We want to obtain POOLING layer configuration'
                    return tf.subtract(1.0, tf.divide(1.0, sum))
            return tf.sigmoid(bias)

        'Computing VISIBLE layer with HIDDEN layer given'
        'If gaussian then return the mean of the normal distribution from wich visible unit are drawn, covariance matrix being self.gaussian_variance square identity'
        'If not gaussian then return the binary probability wich is sigmoid'
        if method == 'backward':
            if self.padding:
                raise Exception('12332323')
                conv = tf.nn.conv2d(operand, self._get_flipped_kernel(), [1, self.up_stride, 1, 1], padding='SAME')
            else:
                conv = tf.nn.conv2d_transpose(
                    operand,
                    self.kernels,
                    (self.batch_size, sequence_length, self.visible_width, self.visible_channels),#self.filter_number
                    [1, self.up_stride, 1, 1],
                    data_format="NHWC",
                    padding='SAME')
            if self.gaussian_unit:
                conv = tf.multiply(conv, self.gaussian_variance)
            bias = tf.nn.bias_add(conv, self.biases_V)
            if self.gaussian_unit:
                return bias
            return tf.sigmoid(bias)

    def draw_samples(self: CRBM, mean_activation, method='forward'):
        """INTENT : Draw samples from distribution of specified parameter
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : If FORWARD then samples for HIDDEN layer (BERNOULLI)
                 If BACKWARD then samples for VISIBLE layer (BERNOULLI OR GAUSSIAN if self.gaussian_unit = True)"""

        if self.gaussian_unit and method == 'backward':
            'In this case mean_activation is the mean of the normal distribution, variance being self.variance^2'
            mu = tf.reshape(mean_activation, [-1])
            dist = tfp.distributions.MultivariateNormalDiag(mu, self.sigma)
            samples = dist.sample()
            return tf.reshape(
                samples, [self.batch_size, self.visible_height, self.visible_width, self.visible_channels])
        elif method == 'forward':
            height = self.hidden_height
            width = self.hidden_width
            channels = self.filter_number
        elif method == 'backward':
            height = self.visible_height
            width = self.visible_width
            channels = self.visible_channels

        samples_rand_uniform = tf.random.uniform([self.batch_size, height, width, channels])
        activations = mean_activation
        choices_for_true = tf.ones([self.batch_size, height, width, channels])
        choices_for_false = tf.zeros([self.batch_size, height, width, channels])
        return tf.where(samples_rand_uniform - activations < 0, choices_for_true, choices_for_false)

    def do_contrastive_divergence(self: CRBM, data, n=1, global_step=0):
        """INTENT : Do one step of n-contrastive divergence algorithm for leaning weight and biases of the CRBM
        ------------------------------------------------------------------------------------------------------------------------------------------
        REMARK : global_step is only needed for computing the learning rate ie adjust the decay"""

        V0, Q0, VN, QN = self._do_gibbs_chain(data, n=n)

        'FOR WEIGHT'
        # [vchannels,vheight,vwidth,batch_size] 1,120,1,1 conv with [hiddenhei,hiddenwidth,batch_size,hiddenchannel] 109,1,1,24 give a [vchannels,filterhei,filterwidth,hiddenchannel] filter 1,12,1,24
        if self.padding:
            raise Exception('test')
            positive = tf.nn.conv2d(
                self._get_padded_visible(tf.transpose(V0, perm=[3, 1, 2, 0])),
                tf.transpose(Q0, perm=[1, 2, 0, 3]),
                [1, self.up_stride, 1, 1],
                padding='VALID')
            negative = tf.nn.conv2d(
                self._get_padded_visible(tf.transpose(VN, perm=[3, 1, 2, 0])),
                tf.transpose(QN, perm=[1, 2, 0, 3]),
                [1, self.up_stride, 1, 1], #
                padding='VALID')
        else:
            # Q0_ = tf.transpose(Q0, perm=[1, 0, 2, 3])
            # positive = tf.nn.conv2d(V0, Q0_, [1, tf.divide(sequence_length, self.filter_height), 1, 1], padding='SAME'); print(positive.shape)
            # negative = tf.nn.conv2d(V0, Q0_, [1, tf.divide(sequence_length, self.filter_height), 1, 1], padding='SAME'); print(negative.shape)
            # print(self.hidden_height)
            # print(tf.divide(sequence_length, self.filter_height))
            positive = tf.nn.conv2d(
                tf.transpose(V0, perm=[3, 1, 2, 0]),  # 1, 120, 1, 1   [vchannels,vheight,vwidth,batch_size]  batch_shape + [in_height, in_width, in_channels
                tf.transpose(Q0, perm=[1, 2, 0, 3]),  # 10, 1, 1, 24   [hiddenhei,hiddenwidth,batch_size,hiddenchannel] [filter_height * filter_width * in_channels, output_channels]
                                                        # (1,12,1,24),[vchannels,filterhei,filterwidth,hiddenchannel] [batch, out_height, out_width, filter_height * filter_width * in_channels].
                [1, tf.divide(sequence_length, self.filter_height), 1, 1],#self.up_stride 10
                padding='SAME')
            negative = tf.nn.conv2d(
                tf.transpose(VN, perm=[3, 1, 2, 0]),  # 1, 120, 1, 1
                tf.transpose(QN, perm=[1, 2, 0, 3]),  # 10, 1, 1, 24
                [1, tf.divide(sequence_length, self.filter_height), 1, 1],#self.up_stride
                padding='SAME')
        ret = positive - negative

        if self.gaussian_unit:
            ret = tf.divide(ret, self.gaussian_variance)
        g_weight = tf.divide(tf.transpose(ret, perm=[1, 2, 0, 3]), self.batch_size)
        g_weight_sparsity = self._get_param_sparsity_penalty('kernel', Q0, V0)
        g_weight_l2 = self._get_param_weight_penalty(self.kernels)

        'FOR BIAS VISIBLE'
        g_biais_V = tf.divide(tf.reduce_sum(tf.subtract(V0, VN), [0, 1, 2]), self.batch_size)
        if self.gaussian_unit:
            g_biais_V = tf.divide(g_biais_V, self.gaussian_variance * self.gaussian_variance)

        'FOR BIAS HIDDEN'
        g_biais_H = tf.divide(tf.reduce_sum(tf.subtract(Q0, QN), [0, 1, 2]), self.batch_size)
        g_biais_H_sparsity = self._get_param_sparsity_penalty('hidden_bias', Q0, V0)

        'UPDATE ALL'
        ret_bv = self._apply_grad(self.biases_V, g_biais_V, self.vitesse_biases_V, self.learning_rate,
                                  global_step=global_step)
        ret_bh = self._apply_grad(self.biases_H, g_biais_H, self.vitesse_biases_H, self.learning_rate, sparsity=True,
                                  sparsity_value=g_biais_H_sparsity, global_step=global_step)
        ret_w = self._apply_grad(self.kernels, g_weight, self.vitesse_kernels, self.learning_rate_w, wd=False, wd_value=g_weight_l2, sparsity=False,
            sparsity_value=g_weight_sparsity, global_step=global_step)

        cost = tf.reduce_sum(tf.square(tf.subtract(data, VN)))
        update = tf.reduce_sum(VN)
        return ret_w, ret_bv, ret_bh, cost, update

    def _do_gibbs_chain(self: CRBM, data, n=1):
        """INTENT : Do one chain of length n starting from input and yields V0 Q0 VN QN in order to compute gradient
        ------------------------------------------------------------------------------------------------------------------------------------------
        """

        V0 = data
        Q0 = self.infer_probability(data, 'forward')
        ret = self.draw_samples(Q0, 'forward')
        'ret is HO'
        for _ in range(n):
            VN = self.draw_samples(self.infer_probability(ret, 'backward'), 'backward')
            QN = self.infer_probability(VN, 'forward')
            ret = self.draw_samples(QN, 'forward')
            'At the end ret is HN'
        return V0, Q0, VN, QN

    def _apply_grad(
            self: CRBM, parameter, grad_value, vitesse, lr, wd=False, wd_value=None, sparsity=False, sparsity_value=None,
            global_step=0):
        """INTENT : Apply gradient descent to provided variable. Also modify it vitesse
        ------------------------------------------------------------------------------------------------------------------------------------------
        """

        # lr = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_step,
        #                                 self.learning_rate_decay, staircase=True)
        ret = tf.add(0.0, tf.multiply(lr, grad_value))
        ret = vitesse.assign(ret)
        if wd:
            ret = tf.subtract(ret, tf.multiply(lr, wd_value))
        if sparsity:
            ret = tf.subtract(ret, tf.multiply(lr, sparsity_value))
        return parameter.assign_add(ret)

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
                    [1, self.up_stride, 1, 1],
                    padding='VALID')
            else:
                retBis = tf.nn.conv2d(
                    tf.transpose(V0, perm=[3, 1, 2, 0]),
                    tf.transpose(baseline, perm=[1, 2, 0, 3]),
                    [1, tf.divide(sequence_length,self.filter_height), 1, 1], #10
                    padding='SAME')
            retBis = tf.transpose(retBis, perm=[1, 2, 0, 3])
            return tf.multiply(ret, retBis)

    def _get_param_weight_penalty(self: CRBM, operand):
        """INTENT : Compute l2 regularization pernalty term
        ------------------------------------------------------------------------------------------------------------------------------------------
        PARAMETERS :
        operand               :        parameter to be regularized"""

        # return tf.multiply(self.weight_decay, operand)
        return tf.multiply(self.weight_decay, operand)

    def _get_flipped_kernel(self):
        """INTENT : Not only flip kernel horizontally and vertically but also swap in and out channels
        ------------------------------------------------------------------------------------------------------------------------------------------
        """

        return tf.reverse(self.kernels, [0, 1])
        # return tf.transpose(tf.reverse(self.kernels, [0, 1]), perm=[0, 1, 3, 2])
        # return tf.transpose(tf.reverse(self.kernels, [True, True, False, False]), perm=[0, 1, 3, 2])

    def _get_padded_hidden(self, hidden):
        """INTENT : Add padding to the hidden layer so that it can be convolved with flipped kernel and give the same size as input
        ------------------------------------------------------------------------------------------------------------------------------------------
        """

        return tf.pad(
                hidden,
                tf.constant(
                    ([0, 0],
                    [self.filter_height - 1, self.filter_height - 1],
                    [self.filter_width - 1, self.filter_width - 1],
                    [0, 0])
             )
        )

    def _get_padded_visible(self: CRBM, visible):
        """INTENT : Add padding to the visible layer so that it can be convolved with hidden layer to compute weight gradient update
        ------------------------------------------------------------------------------------------------------------------------------------------
        """

        return tf.pad(
            visible,
            [[0, 0],
             [np.floor((self.filter_height - 1) / 2).astype(int),
              np.ceil((self.filter_height - 1) / 2).astype(int)],
             [np.floor((self.filter_width - 1) / 2).astype(int),
              np.ceil((self.filter_width - 1) / 2).astype(int)],
             [0, 0]])

if __name__ == "__main__":
    should_scale_down = False
    should_plot_training_substeps = False

    batch_size = 1
    sequence_length = 48
    filter_length = 12
    n_filters = 24

    num_epochs = 28
    ###
    x_train, x_test = read_and_preprocess_data(
        sequence_length=sequence_length,
        batch_size=batch_size,
        motes_train=[7],
        motes_test=[7]
    )
    inp = None
    block = np.divide(np.arange(-int(6), int(6)), 6) if should_scale_down else np.arange(-int(6), int(6))

    for _ in range(int(sequence_length/filter_length)):
        if inp is None:
            inp = block
        else:
            inp = np.concatenate((inp, block))

    mean = np.mean(inp)
    inputs = tf.cast(tf.reshape(tf.constant(inp), (1, sequence_length, 1, 1)), tf.float32)

    c = CRBM(
        'crbm',
        f_number=n_filters, batch_size=batch_size,
        f_height=filter_length, f_width=1,
        up_stride=filter_length, padding=False,
        v_height=sequence_length, v_width=1, v_channels=1,
        init_biases_H = -3.81,
        init_biases_V = mean
    )

    def predict_and_plot(inp, name):
        p1 = c.infer_probability(inp, 'forward')
        a1 = c.draw_samples(p1)

        pv1 = c.infer_probability(a1, 'backward')
        av1 = c.draw_samples(pv1, 'backward')

        plt.plot(inp.numpy().reshape(-1), label='original')
        plt.plot(av1.numpy().reshape(-1), label='reconstruction')
        plt.legend()
        harry(name)
        return per_rms_diff(inp, av1)
    prms_before = predict_and_plot(inputs, 'before')

    old_weights, old_h_bias, old_v_bias = tf.identity(c.kernels), tf.identity(c.biases_H), tf.identity(c.biases_V)
    # Training for some epochs
    losses = []
    for y in range(num_epochs):
        _, _, _, cost, _ =c.do_contrastive_divergence(inputs)
        losses.append(cost)
        if should_plot_training_substeps:
            if y % int(num_epochs/10) == 0: predict_and_plot(inputs, f'{y}')
    plt.plot(losses)
    harry('learning_curve')

    new_weights, new_h_bias, new_v_bias = tf.identity(c.kernels), tf.identity(c.biases_H), tf.identity(c.biases_V)

    _, ax = plt.subplots(n_filters, figsize=(15,50))
    for x in range(n_filters):
        old = old_weights.numpy()[:, :, :, x].reshape(-1)
        new = new_weights.numpy()[:, :, :, x].reshape(-1)
        ax[x].plot(old, label=f'old_{x}')
        ax[x].plot(new, label=f'new_{x}')
        ax[x].legend()
    harry('weights')
    plt.figure(figsize=(8, 6))
    # Checking performance afterwards
    prms_after = predict_and_plot(inputs, 'after')
    print(f'PRMS from {prms_before} -> {prms_after}')
