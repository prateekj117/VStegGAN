import tensorflow as tf

def feature_network(secret, weights):
    layers = []

    with tf.variable_scope('f_net'):
      net = tf.layers.conv3d(inputs=secret, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[0]),
                           bias_initializer=tf.constant_initializer(weights[1]))
      #print("1: ",net) => [8x320x240x64]
      layers.append(net)

      net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')
      #print("2: ",net) => [8x160x120x64]

      net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[2]),
                            bias_initializer=tf.constant_initializer(weights[3]))
      #print("3: ",net) => [8x160x240x128]
      layers.append(net)

      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
      #print("4: ",net) => [4x80x60x128]

      net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[4]),
                            bias_initializer=tf.constant_initializer(weights[5]))
      net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[6]),
                            bias_initializer=tf.constant_initializer(weights[7]))
      #print("5: ",net) => [4x80x60x256]
      layers.append(net)

      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
      #print("6: ",net) => [2x40x30x256]

      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[8]),
                            bias_initializer=tf.constant_initializer(weights[9]))
      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=tf.constant_initializer(weights[10]),
                            bias_initializer=tf.constant_initializer(weights[11]))
      #print("7: ",net) => [2x40x30x512]
      layers.append(net)

      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
      #print("8: ",net) => [1x20x15x512]

      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
      #print("9: ",net) => [3x22x17x512]
      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                            kernel_initializer=tf.constant_initializer(weights[12]),
                            bias_initializer=tf.constant_initializer(weights[13]))
      #print("10: ",net) => [1x20x15x512]

      latent_space = tf.identity(net)

      net = tf.keras.layers.UpSampling3D(size=2)(net)
      #print("11: ",net) => [2x40x30x512]

      net = tf.concat([net, layers.pop()], axis=-1)
      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
      #print("12: ",net) => [2x40x30x512]

      net = tf.keras.layers.UpSampling3D(size=2)(net)
      #print("13: ",net) => [4x80x60x512]

      net = tf.concat([net, layers.pop()], axis=-1)
      net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
      net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu)
      #print("14: ",net) => [4x80x60x256]

      net = tf.keras.layers.UpSampling3D(size=2)(net)
      #print("15: ",net) => [8x160x120x256]

      net = tf.concat([net, layers.pop()], axis=-1)
      net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)
      #print("16: ",net) => [8x160x120x128]

      net = tf.keras.layers.UpSampling3D(size=(1,2,2))(net)
      #print("17: ",net) => [8x320x240x128]

      net = tf.concat([net, layers.pop()], axis=-1)
      net = tf.layers.conv3d(inputs=net, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)
      #print("18: ",net) => [8x320x240x64]

      output = tf.layers.conv3d(net, 3, 3, padding='SAME', activation=tf.nn.sigmoid)
      #print("19: ", output) => [8x320x240x3]

      '''
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
      #print("11: ",net) => [3x22x17x512]

      net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                            kernel_initializer=tf.constant_initializer(weights[14]),
                            bias_initializer=tf.constant_initializer(weights[15]))
      #print("12: ",net) => [1x20x15x512]

      net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
      #print("13: ",net) => [1x10x8x512]

      output = tf.layers.conv3d(net,3,3,padding='same', activation=tf.nn.sigmoid, data_format='channels_last')
      #print("14: ",output) => [1x10x8x3]
      '''

      return latent_space, output


#input_shape = (None, 8, 320, 240, 3)
#secret = tf.placeholder(shape=input_shape, dtype=tf.float32, name='encoded_feature')
#encoded_feature = feature_network(secret, weights)
