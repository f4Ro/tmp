from __future__ import division
import tensorflow as tf
import numpy as np


class CRBM(object):
  """CONVOLUTIONAL RESTRICTED BOLTZMANN MACHINE"""

  def __init__(self, name, v_height = 1, v_width = 1, v_channels = 784, f_height = 1, f_width = 1, f_number = 400,
               init_biases_H = -3, init_biases_V = 0.01, init_weight_stddev = 0.01,
               gaussian_unit = True, gaussian_variance = 0.2,
               prob_maxpooling = False,
               batch_size = 20, learning_rate = 0.0001, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1):
    """INTENT : Initialization of a Convolutional Restricted Boltzmann Machine
    ------------------------------------------------------------------
    REMARK : This class can represent differents kinds of RBM
             fully connected RBM              if  fully_connected = True and then v_height=v_width=f_height=f_width=1
             gaussian RBM                     if  gaussian_unit = True     (https://web.eecs.umich.edu/~honglak/iccv2011-sparseConvLearning.pdf)
             sparse   RBM                     if  sparsity_coef <> 0        (http://ai.stanford.edu/~ang/papers/nips07-sparsedeepbeliefnetworkv2.pdf   -    https://www.ii.pwr.edu.pl/~gonczarek/papers/rbm_sparse.pdf)
             probabilistic maxpooling CRBM    if  prob_maxpooling = True   (http://www.cs.toronto.edu/~rgrosse/icml09-cdbn.pdf)"""

    try:
      self.name                  = name
      self.visible_height        = v_height
      self.visible_width         = v_width
      self.visible_channels      = v_channels
      self.filter_height         = f_height
      self.filter_width          = f_width
      self.filter_number         = f_number
      self.gaussian_unit         = gaussian_unit
      if gaussian_unit:
        self.gaussian_variance   =     gaussian_variance
      self.prob_maxpooling       =     prob_maxpooling
      self.hidden_height         =     v_height - f_height + 1
      self.hidden_width          =     v_width - f_width + 1

      self.batch_size            =     batch_size
      self.learning_rate         =     learning_rate
      self.learning_rate_decay   =     learning_rate_decay
      self.momentum              =     momentum
      self.decay_step            =     decay_step
      self.weight_decay          =     weight_decay
      self.sparsity_target       =     sparsity_target
      self.sparsity_coef         =     sparsity_coef

      with tf.variable_scope(name):
         with tf.device('/cpu:0'):
          self.kernels            = tf.get_variable('weights', (f_height, f_width,v_channels,f_number), initializer=tf.truncated_normal_initializer(stddev=init_weight_stddev, dtype=tf.float32), dtype=tf.float32)
          self.biases_V           = tf.get_variable('biases_V', (v_channels), initializer=tf.constant_initializer(init_biases_V), dtype=tf.float32)
          self.biases_H           = tf.get_variable('biases_H', (f_number), initializer=tf.constant_initializer(init_biases_H), dtype=tf.float32)
          self.vitesse_kernels    = tf.get_variable('vitesse_weights', (f_height, f_width,v_channels,f_number), initializer=tf.constant_initializer(0), dtype=tf.float32)
          self.vitesse_biases_V   = tf.get_variable('vitesse_biases_V', (v_channels), initializer=tf.constant_initializer(0), dtype=tf.float32)
          self.vitesse_biases_H   = tf.get_variable('vitesse_biases_H', (f_number), initializer=tf.constant_initializer(0), dtype=tf.float32)
          if gaussian_unit:
            self.sigma              = tf.get_variable('sigma', (v_width*v_height*batch_size*v_channels), initializer=tf.constant_initializer(self.gaussian_variance * self.gaussian_variance), dtype = tf.float32)

    except ValueError as error:
      print('--------------------------')
      print(error.args)
      print('--------------------------')


  def compute_energy(self, visible, hidden, method = 'forward'):
    """INTENT : Compute the energy of the configuration (visible,hidden) given in parameters
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : The returned result is per configuration so does not depend on the batch_size"""

    if method == 'forward':
      conv = tf.nn.conv2d(visible, self.kernels, [1, 1, 1, 1], padding='VALID')
      operand = hidden
    elif method == 'backward':
      conv = tf.nn.conv2d(self._get_padded_hidden(hidden), self._get_flipped_kernel(), [1, 1, 1, 1], padding='VALID')
      operand = visible
    weight =  tf.reduce_sum(tf.mul(operand,conv))
    bias_H =  tf.reduce_sum(tf.mul(self.biases_H,tf.reduce_sum(hidden,[0,1,2])))

    if self.gaussian_unit:
      'GAUSSIAN UNIT IN VISIBLE LAYER'
      weight = tf.div(weight,self.gaussian_variance)
      bias_V = tf.reduce_sum(tf.square(tf.sub(visible,tf.reshape(self.biases_V,[1,1,1,self.visible_channels]))))
      bias_V = tf.div(bias_V,2*self.gaussian_variance*self.gaussian_variance)
      output = tf.sub(bias_V,tf.add(bias_H,weight))
    else:
      'BINARY UNIT IN VISIBLE LAYER'
      bias_V =  tf.reduce_sum(tf.mul(self.biases_V,tf.reduce_sum(visible,[0,1,2])))
      output =  tf.mul(-1,tf.add(weight,tf.add(bias_H,bias_V)))
    return tf.div(output,self.batch_size)


  def infer_probability(self, operand, method, result = 'hidden'):
    """INTENT : Compute the probabily of activation of one layer given the other
    REMARK : If method is FORWARD then compute hidden layer probability and operand is VISIBLE layer
             If method is BACKWARD then compute visible layer probability and operand is HIDDEN layer"""

    'Computing HIDDEN layer with VISIBLE layer given'
    'Gaussian visible or not, hidden layer activation is a sigmoid'
    if method == 'forward':
      conv = tf.nn.conv2d(operand, self.kernels, [1, 1, 1, 1], padding='VALID')
      if self.gaussian_unit:
        conv = tf.div(conv,self.gaussian_variance)
      bias = tf.nn.bias_add(conv, self.biases_H)
      if self.prob_maxpooling:
        'SPECIFIC CASE where we enable probabilistic max pooling'
        exp = tf.exp(bias)
        custom_kernel = tf.constant(1.0, shape=[2,2,self.filter_number,1])
        sum = tf.nn.depthwise_conv2d(exp, custom_kernel, [1, 2, 2, 1], padding='VALID')
        sum = tf.add(1.0,sum)
        ret_kernel = np.zeros((2,2,self.filter_number,self.filter_number))
        for i in range(2):
          for j in range(2):
            for k in range(self.filter_number):
              ret_kernel[i,j,k,k] = 1
        custom_kernel_bis = tf.constant(ret_kernel,dtype = tf.float32)
        sum_bis =  tf.nn.conv2d_transpose(sum, custom_kernel_bis, (self.batch_size,self.hidden_height,self.hidden_width,self.filter_number), strides= [1, 2, 2, 1], padding='VALID', name=None)
        if result == 'hidden':
          'We want to obtain HIDDEN layer configuration'
          return tf.div(exp,sum_bis)
        elif result == 'pooling':
          'We want to obtain POOLING layer configuration'
          return tf.sub(1.0,tf.div(1.0,sum))
      return tf.sigmoid(bias)

    'Computing VISIBLE layer with HIDDEN layer given'
    'If gaussian then return the mean of the normal distribution from wich visible unit are drawn, covariance matrix being self.gaussian_variance square identity'
    'If not gaussian then return the binary probability wich is sigmoid'
    if method == 'backward':
      conv = tf.nn.conv2d(self._get_padded_hidden(operand), self._get_flipped_kernel(), [1, 1, 1, 1], padding='VALID')
      if self.gaussian_unit:
        conv = tf.mul(conv,self.gaussian_variance)
      bias = tf.nn.bias_add(conv, self.biases_V)
      if self.gaussian_unit:
        return bias
      return tf.sigmoid(bias)


  def draw_samples(self, mean_activation, method ='forward'):
    """INTENT : Draw samples from distribution of specified parameter
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : If FORWARD then samples for HIDDEN layer (BERNOULLI)
             If BACKWARD then samples for VISIBLE layer (BERNOULLI OR GAUSSIAN if self.gaussian_unit = True)"""

    if self.gaussian_unit and method == 'backward':
      'In this case mean_activation is the mean of the normal distribution, variance being self.variance^2'
      mu = tf.reshape(mean_activation, [-1])
      dist = tf.contrib.distributions.MultivariateNormalDiag(mu, self.sigma)
      samples = dist.sample()
      return tf.reshape(samples,[self.batch_size,self.visible_height,self.visible_width,self.visible_channels])
    elif method == 'forward':
      height   =  self.hidden_height
      width    =  self.hidden_width
      channels =  self.filter_number
    elif method == 'backward':
      height   =  self.visible_height
      width    =  self.visible_width
      channels =  self.visible_channels
    return tf.select(tf.random_uniform([self.batch_size,height,width,channels]) - mean_activation < 0, tf.ones([self.batch_size,height,width,channels]), tf.zeros([self.batch_size,height,width,channels]))


  def do_contrastive_divergence(self, data, n = 1, global_step = 0):
    """INTENT : Do one step of n-contrastive divergence algorithm for leaning weight and biases of the CRBM
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : global_step is only needed for computing the learning rate ie adjust the decay"""

    V0,Q0,VN,QN = self._do_gibbs_chain(data, n = n)

    'FOR WEIGHT'
    #[vchannels,vheight,vwidth,batch_size] conv with [hiddenhei,hiddenwidth,batch_size,hiddenchannel] give a [vchannels,filterhei,filterwidth,hiddenchannel] filter
    positive = tf.nn.conv2d(tf.transpose(V0,perm=[3,1,2,0]), tf.transpose(Q0,perm=[1,2,0,3]), [1, 1, 1, 1], padding='VALID')
    negative = tf.nn.conv2d(tf.transpose(VN,perm=[3,1,2,0]), tf.transpose(QN,perm=[1,2,0,3]), [1, 1, 1, 1], padding='VALID')
    ret = positive - negative
    if self.gaussian_unit:
      ret = tf.div(ret,self.gaussian_variance)
    g_weight          = tf.div(tf.transpose(ret,perm=[1,2,0,3]),self.batch_size)
    g_weight_sparsity = self._get_param_sparsity_penalty('kernel', Q0, V0)
    g_weight_l2       = self._get_param_weight_penalty(self.kernels)

    'FOR BIAS VISIBLE'
    g_biais_V = tf.div(tf.reduce_sum(tf.sub(V0,VN), [0,1,2]),self.batch_size)
    if self.gaussian_unit:
      g_biais_V = tf.div(g_biais_V,self.gaussian_variance * self.gaussian_variance)

    'FOR BIAS HIDDEN'
    g_biais_H            =   tf.div(tf.reduce_sum(tf.sub(Q0,QN), [0,1,2]),self.batch_size)
    g_biais_H_sparsity   =   self._get_param_sparsity_penalty('hidden_bias', Q0, V0)

    'UPDATE ALL'
    ret_w  = self._apply_grad(self.kernels,  g_weight,  self.vitesse_kernels, wd = True, wd_value = g_weight_l2, sparsity = True, sparsity_value = g_weight_sparsity,  global_step = global_step)
    ret_bv = self._apply_grad(self.biases_V, g_biais_V, self.vitesse_biases_V,                                                                                         global_step = global_step)
    ret_bh = self._apply_grad(self.biases_H, g_biais_H, self.vitesse_biases_H,                                   sparsity = True, sparsity_value = g_biais_H_sparsity, global_step = global_step)
    cost = tf.reduce_sum(tf.square(tf.sub(data,VN)))
    update = tf.reduce_sum(VN)
    return ret_w, ret_bv, ret_bh, cost, update


  def save_parameter(self, path, sess, step):
    """INTENT : Save parameter of the RBM
    ------------------------------------------------------------------------------------------------------------------------------------------

    """
    saver = tf.train.Saver([self.kernels, self.biases_V, self.biases_H])
    return saver.save(sess, path, global_step=step)


  def load_parameter(self, path, sess):
    """INTENT : Load parameter of this RBM if the save file exist
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    path               :        where is the save file
    sess               :        the session = context we want to load the parameter"""

    saver = tf.train.Saver([self.kernels, self.biases_V, self.biases_H])
    return saver.restore(sess, path)


  def init_parameter(self,from_scratch = True):
    """INTENT : Return the tensorflow operation for initializing the parameter of this RBM
    ------------------------------------------------------------------------------------------------------------------------------------------
    """

    if from_scratch:
      return tf.initialize_all_variables()
    elif self.gaussian_unit:
      return tf.initialize_variables([self.vitesse_kernels, self.vitesse_biases_V, self.vitesse_biases_H, self.sigma])
    else:
      return tf.initialize_variables([self.vitesse_kernels, self.vitesse_biases_V, self.vitesse_biases_H])


  def _do_gibbs_chain(self, data, n = 1):
    """INTENT : Do one chain of length n starting from input and yields V0 Q0 VN QN in order to compute gradient
    ------------------------------------------------------------------------------------------------------------------------------------------
    """

    V0 = data
    Q0 = self.infer_probability(data,'forward')
    ret = self.draw_samples(Q0, 'forward')
    'ret is HO'
    for i in range(n):
      VN = self.draw_samples(self.infer_probability(ret,'backward'), 'backward')
      QN = self.infer_probability(VN,'forward')
      ret = self.draw_samples(QN, 'forward')
      'At the end ret is HN'
    return V0,Q0,VN,QN


  def _apply_grad(self, parameter, grad_value, vitesse, wd = False, wd_value = None, sparsity = False, sparsity_value = None, global_step = 0):
    """INTENT : Apply gradient descent to provided variable. Also modify it vitesse
    ------------------------------------------------------------------------------------------------------------------------------------------
    """

    lr = tf.train.exponential_decay(self.learning_rate,global_step,self.decay_step,self.learning_rate_decay,staircase=True)
    ret = tf.add(tf.mul(self.momentum,vitesse),tf.mul(lr,grad_value))
    ret = vitesse.assign(ret)
    if wd:
      ret = tf.sub(ret,tf.mul(lr,wd_value))
    if sparsity:
      ret = tf.sub(ret,tf.mul(lr,sparsity_value))
    return parameter.assign_add(ret)


  def _get_param_sparsity_penalty(self, name, Q0, V0):
    """INTENT : Compute sparsity penalty term (l2 norm of target minus mean activation)
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : see http://ai.stanford.edu/~ang/papers/nips07-sparsedeepbeliefnetworkv2.pdf
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : formula is for hidden_bias: -2lambda*sumonwidth/height[meanonbatch[(1-Q0)Q0(target-meanonbatch[Q0])]]
                        for kernel      -2lambda*sumonwidth/height[meanonbatch[(1-Q0)Q0(target-meanonbatch[Q0])v]]     vi+r-1,j+s-1,(l) = v"""

    ret = -2*self.sparsity_coef/self.batch_size
    if self.gaussian_unit:
      ret = ret/self.gaussian_variance
    mean = tf.reduce_sum(Q0, [0], keep_dims = True)
    baseline = tf.mul(tf.sub(self.sparsity_target,mean),tf.mul(tf.sub(1.0,Q0),Q0))
    if name == 'hidden_bias':
      return tf.mul(ret,tf.reduce_sum(baseline, [0,1,2]))
    if name == 'kernel':
      retBis = tf.nn.conv2d(tf.transpose(V0,perm=[3,1,2,0]), tf.transpose(baseline,perm=[1,2,0,3]), [1, 1, 1, 1], padding='VALID')
      retBis = tf.transpose(retBis,perm=[1,2,0,3])
      return tf.mul(ret,retBis)


  def _get_param_weight_penalty(self, operand):
    """INTENT : Compute l2 regularization pernalty term
    ------------------------------------------------------------------------------------------------------------------------------------------
    """

    return tf.mul(self.weight_decay,operand)


  def _get_flipped_kernel(self):
    """INTENT : Not only flip kernel horizontally and vertically but also swap in and out channels
    ------------------------------------------------------------------------------------------------------------------------------------------
    """

    return tf.transpose(tf.reverse(self.kernels,[True,True,False,False]),perm=[0, 1, 3, 2])


  def _get_padded_hidden(self, hidden):
    """INTENT : Add padding to the hidden layer so that it can be convolved with flipped kernel and give the same size as input
    ------------------------------------------------------------------------------------------------------------------------------------------
    """

    return tf.pad(hidden,[[0,0],[self.filter_height-1,self.filter_height-1],[self.filter_width-1,self.filter_width-1],[0,0]])


  def _get_padded_visible(self, visible):
    """INTENT : Add padding to the visible layer so that it can be convolved with hidden layer to compute weight gradient update
    ------------------------------------------------------------------------------------------------------------------------------------------
    """

    return tf.pad(visible,[[0,0],[np.floor((self.filter_height-1)/2).astype(int),np.ceil((self.filter_height-1)/2).astype(int)],[np.floor((self.filter_width-1)/2).astype(int),np.ceil((self.filter_width-1)/2).astype(int)],[0,0]])


if __name__ == '__main__':
    batch_size = 32
    c = CRBM(
        'crbm',
        v_height=120, v_width=1, v_channels=1,
        f_height=12, f_width=1,
        f_number=24,
        batch_size=batch_size
    )
    print('Running on tf version: ', tf.__version__)
    test_inputs = tf.random_normal((batch_size, 120, 1, 1))

    def run_n_times(n: int) -> None:
        curr_loss = float("inf")
        for _ in range(n):
            curr_loss = c.do_contrastive_divergence(test_inputs)[3]
        return curr_loss
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print(sess.run(run_n_times(100)))
