import tensorflow as tf


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1,1],[1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv3d(padded_input, out_channels, kernel_size=3, strides=stride, padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=-1, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def discriminating_network(discrim_inputs, ndf=64):
        n_layers = 3
        layers = []

        # input: [batch_size, frames_per_batch, img_width, img_height, in_channels]
        #here: batch_size=1, frames_per_batch=8, img_width=240, img_height=320, in_channels=3
        input = tf.identity(discrim_inputs)


        # layer_1: [batch, 8, 320, 240, 3] => [batch, 8, 160, 120, 64]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, ndf, stride=(1,2,2))
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)


        # layer_2: [batch, 8, 160, 120, 64] => [batch, 4, 80, 60, 128]
        # layer_3: [batch, 4, 80, 60, 128] => [batch, 2, 40, 30, 256]
        # layer_4: [batch, 2, 40, 30, 256] => [batch, 1, 20, 15, 512]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2**(i+1), 8)
                convolved = discrim_conv(layers[-1], out_channels, stride=2)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)


        # layer_5: [batch, 1, 20, 15, 512] => [batch, 1, 20, 15, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

#input_shape = (None,8,320,240,3)
#discrim_input = tf.placeholder(shape=input_shape, dtype=tf.float32, name='discrim_input')
#prob = discriminating_network(discrim_input, 64)
#print(prob)
