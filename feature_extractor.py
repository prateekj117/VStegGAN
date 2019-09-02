import tensorflow as tf


def conv3d(name, l_input, w, b):
  return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def feature_network(_X, _dropout, batch_size, _weights, _biases):
    with tf.variable_scope('f_net'):
        # Convolution Layer
        conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = max_pool('pool1', conv1, k=1)

        # Convolution Layer
        conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = max_pool('pool2', conv2, k=2)

        # Convolution Layer
        conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = max_pool('pool3', conv3, k=2)

        # Convolution Layer
        conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = max_pool('pool4', conv4, k=2)

        # Convolution Layer
        conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = max_pool('pool5', conv5, k=2)

        output = tf.layers.conv3d(pool5,3,3,padding='same',activation=tf.nn.sigmoid,data_format='channels_last')

        return out
