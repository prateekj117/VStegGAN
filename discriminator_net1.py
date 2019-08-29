import tensorflow as tf

def discriminator_network(cover, container):
    with tf.variable_scope('d_net'):
        concat_input = tf.concat([cover, secret],  axis = 1, name='concat')
        #print(concat_input) 16x320x240x3

        conv1_1 = tf.layers.conv3d(concat_input,32,3,padding='same',name="1",activation=tf.nn.relu,data_format='channels_last')  
        conv1_2 = tf.layers.conv3d(conv1_1,32,3,padding='same',name="2",activation=tf.nn.relu,data_format='channels_last') 
        conv1_3 = tf.layers.conv3d(conv1_2,32,3,padding='same',name="3",activation=tf.nn.relu,data_format='channels_last')
        
        convmax = tf.layers.max_pooling3d(conv1_3, (2,1,1), strides=(2,1,1), data_format='channels_last')
        conv1 = tf.layers.conv3d(convmax,32,3,padding='same',name="4",activation=tf.nn.relu,data_format='channels_last')
        
        #print(conv1) 16x320x240x3

        maxpool1 = tf.layers.max_pooling3d(conv1_3, 2, strides=2, data_format='channels_last')
        # print(maxpool1) 8x160x120x32

        conv2_1 = tf.layers.conv3d(maxpool1,64,3,padding='same',name="5",activation=tf.nn.relu,data_format='channels_last')  
        conv2_2 = tf.layers.conv3d(conv2_1,64,3,padding='same',name="6",activation=tf.nn.relu,data_format='channels_last') 
        conv2_3 = tf.layers.conv3d(conv2_2,64,3,padding='same',name="7",activation=tf.nn.relu,data_format='channels_last')

        conv2 = tf.layers.conv3d(conv2_3,64,3,padding='same',name="8",activation=tf.nn.relu,data_format='channels_last') 

        maxpool2 = tf.layers.max_pooling3d(conv2_3, 2, strides=2, data_format='channels_last')
        # print(maxpool2) 4x80x60x64
        
        conv3_1 = tf.layers.conv3d(maxpool2,128,3,padding='same',name="9",activation=tf.nn.relu,data_format='channels_last')  
        conv3_2 = tf.layers.conv3d(conv3_1,128,3,padding='same',name="10",activation=tf.nn.relu,data_format='channels_last') 
        conv3_3 = tf.layers.conv3d(conv3_2,128,3,padding='same',name="11",activation=tf.nn.relu,data_format='channels_last')

        conv3 = tf.layers.conv3d(conv3_3,128,3,padding='same',name="12",activation=tf.nn.relu,data_format='channels_last') 
    
        maxpool3 = tf.layers.max_pooling3d(conv3_3, 2, strides=2, data_format='channels_last')
        # print(maxpool3) 2x40x30x128
        
        conv4_1 = tf.layers.conv3d(maxpool3,256,3,padding='same',name="13",activation=tf.nn.relu,data_format='channels_last')  
        conv4_2 = tf.layers.conv3d(conv4_1,256,3,padding='same',name="14",activation=tf.nn.relu,data_format='channels_last') 
        conv4_3 = tf.layers.conv3d(conv4_2,256,3,padding='same',name="15",activation=tf.nn.relu,data_format='channels_last')
        
        conv4 = tf.layers.conv3d(conv4_3,256,3,padding='same',name="16",activation=tf.nn.relu,data_format='channels_last')
        #print(conv4)     
       
        maxpool4 = tf.layers.max_pooling3d(conv4_3, (1,2,2), strides=(1,2,2), data_format='channels_last')
        # print(maxpool4) 2x20x15x256

        
        print(output)
        return output