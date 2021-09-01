import tensorflow as tf

inputs = tf.zeros((1, 120, 1, 24))  # batch_shape + [in_height, in_width, in_channels]
                     # should be 1, 120, 1, 24
hiddens = tf.zeros((1, 10 , 1, 24)) # [filter_height, filter_width, in_channels, out_channels]
                     # should be 10, 1, 1, 24
hiddens = tf.transpose(hiddens, perm=[1, 0, 2, 3])

print(tf.nn.conv2d(inputs, hiddens, [1, 12, 1, 1], padding='SAME').shape) # [batch, out_height, out_width, filter_height * filter_width * in_channels]
