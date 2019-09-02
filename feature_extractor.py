import tensorflow as tf

def feature_network(secret):
    with tf.variable_scope('f_net'):
      net = tf.layers.conv3d(inputs=secret, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[0]),
                           bias_initializer=tf.constant_initializer(weights[1]))
      net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

      net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[2]),
                            bias_initializer=tf.constant_initializer(weights[3]))
      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

      net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[4]),
                            bias_initializer=tf.constant_initializer(weights[5]))
      net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[6]),
                            bias_initializer=tf.constant_initializer(weights[7]))
      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[8]),
                            bias_initializer=tf.constant_initializer(weights[9]))
      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[10]),
                            bias_initializer=tf.constant_initializer(weights[11]))
      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])

      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                            kernel_initializer=tf.constant_initializer(weights[12]),
                            bias_initializer=tf.constant_initializer(weights[13]))
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                            kernel_initializer=tf.constant_initializer(weights[14]),
                            bias_initializer=tf.constant_initializer(weights[15]))
      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

      output = tf.layers.conv3d(net,3,3,padding='same',activation=tf.nn.sigmoid,data_format='channels_last')

      return out
