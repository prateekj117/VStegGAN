import tensorflow as tf

def revealing_network(container):
    with tf.variable_scope('r_net'):

        conv1_1 = tf.layers.conv3d(container,32,3,padding='same',name="1",activation=tf.nn.relu,data_format='channels_last')  
        conv1_2 = tf.layers.conv3d(conv1_1,32,3,padding='same',name="2",activation=tf.nn.relu,data_format='channels_last') 
        conv1_3 = tf.layers.conv3d(conv1_2,32,3,padding='same',name="3",activation=tf.nn.relu,data_format='channels_last')
        
        conv1 = tf.layers.conv3d(conv1_3,32,3,padding='same',name="4",activation=tf.nn.relu,data_format='channels_last')
        
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


        conv5_1 = tf.layers.conv3d(maxpool4,512,3,padding='same',name="17",activation=tf.nn.relu,data_format='channels_last')  
        conv5_2 = tf.layers.conv3d(conv5_1,512,3,padding='same',name="18",activation=tf.nn.relu,data_format='channels_last') 
        conv5_3 = tf.layers.conv3d(conv5_2,512,3,padding='same',name="19",activation=tf.nn.relu,data_format='channels_last')


        upsample1 = tf.keras.layers.UpSampling3D(size=(1,2,2),  data_format='channels_last')(conv5_3)
        #2x40x30x512
        concat1 = tf.concat([conv4, upsample1], axis = 4, name = 'concat1')
        
        conv6_1 = tf.layers.conv3d(concat1,256,3,padding='same',name="20",activation=tf.nn.relu,data_format='channels_last')  
        conv6_2 = tf.layers.conv3d(conv6_1,256,3,padding='same',name="21",activation=tf.nn.relu,data_format='channels_last') 
        conv6_3 = tf.layers.conv3d(conv6_2,256,3,padding='same',name="22",activation=tf.nn.relu,data_format='channels_last')

        upsample2 = tf.keras.layers.UpSampling3D(size=2,  data_format='channels_last')(conv6_3)
        # print(upsample2) 4x80x60x256

        conv6 = tf.layers.conv3d(conv6_3,256,3,padding='same',name="23",activation=tf.nn.relu,data_format='channels_last')

        concat2 = tf.concat([conv3, upsample2], axis = 4, name = 'concat2')
        
        conv7_1 = tf.layers.conv3d(concat2,128,3,padding='same',name="24",activation=tf.nn.relu,data_format='channels_last')  
        conv7_2 = tf.layers.conv3d(conv7_1,128,3,padding='same',name="25",activation=tf.nn.relu,data_format='channels_last') 
        conv7_3 = tf.layers.conv3d(conv7_2, 128,3,padding='same',name="26",activation=tf.nn.relu,data_format='channels_last')
        

        maxpool5 = tf.layers.max_pooling3d(conv7_3, 2, strides=2, data_format='channels_last')

        conv7 = tf.layers.conv3d(conv7_3,128,3,padding='same',name="27",activation=tf.nn.relu,data_format='channels_last')
        #print(conv7) 2x40x30x128    

        concat3 = tf.concat([conv4, conv6, maxpool5], axis = 4, name = 'concat3')
        
        conv8_1 = tf.layers.conv3d(concat3,256,3,padding='same',name="28",activation=tf.nn.relu,data_format='channels_last')  
        conv8_2 = tf.layers.conv3d(conv8_1,256,3,padding='same',name="29",activation=tf.nn.relu,data_format='channels_last') 
        conv8_3 = tf.layers.conv3d(conv8_2,256,3,padding='same',name="30",activation=tf.nn.relu,data_format='channels_last')

        upsample3 = tf.keras.layers.UpSampling3D(size=2,  data_format='channels_last')(conv8_3)
        # print(upsample3) 4x80x60x256
        
        concat4 = tf.concat([conv3, conv7, upsample3], axis = 4, name = 'concat4')
        
        conv9_1 = tf.layers.conv3d(concat4,128,3,padding='same',name="31",activation=tf.nn.relu,data_format='channels_last')  
        conv9_2 = tf.layers.conv3d(conv9_1,128,3,padding='same',name="32",activation=tf.nn.relu,data_format='channels_last') 
        conv9_3 = tf.layers.conv3d(conv9_2,128,3,padding='same',name="33",activation=tf.nn.relu,data_format='channels_last')

        conv9 = tf.layers.conv3d(conv9_3,128,3,padding='same',name="34",activation=tf.nn.relu,data_format='channels_last')

        upsample4 = tf.keras.layers.UpSampling3D(size=2,  data_format='channels_last')(conv9_3)
        # print(upsample3) 8x160x120x128
        
        concat5 = tf.concat([conv2, upsample4], axis = 4, name = 'concat5')

        conv10_1 = tf.layers.conv3d(concat5,64,3,padding='same',name="35",activation=tf.nn.relu,data_format='channels_last')  
        conv10_2 = tf.layers.conv3d(conv10_1,64,3,padding='same',name="36",activation=tf.nn.relu,data_format='channels_last') 
        conv10_3 = tf.layers.conv3d(conv10_2,64,3,padding='same',name="37",activation=tf.nn.relu,data_format='channels_last')

        maxpool6 = tf.layers.max_pooling3d(conv10_3, 2, strides=2, data_format='channels_last')
        #4x80x60x64

        conv10 = tf.layers.conv3d(conv10_3,64,3,padding='same',name="38",activation=tf.nn.relu,data_format='channels_last')

        concat6 = tf.concat([conv3, conv7, conv9, maxpool6], axis = 4, name = 'concat6')
 	
        conv11_1 = tf.layers.conv3d(concat6,128,3,padding='same',name="39",activation=tf.nn.relu,data_format='channels_last')  
        conv12_2 = tf.layers.conv3d(conv11_1,128,3,padding='same',name="40",activation=tf.nn.relu,data_format='channels_last') 
        conv13_3 = tf.layers.conv3d(conv12_2,128,3,padding='same',name="41",activation=tf.nn.relu,data_format='channels_last')

        upsample5 = tf.keras.layers.UpSampling3D(size=2,  data_format='channels_last')(conv13_3)
        # print(upsample3) 8x160x120x128
        
        concat7 = tf.concat([conv2, conv10, upsample5], axis = 4, name = 'concat7')

        conv12_1 = tf.layers.conv3d(concat7,64,3,padding='same',name="42",activation=tf.nn.relu,data_format='channels_last')  
        conv12_2 = tf.layers.conv3d(conv12_1,64,3,padding='same',name="43",activation=tf.nn.relu,data_format='channels_last') 
        conv12_3 = tf.layers.conv3d(conv12_2,64,3,padding='same',name="44",activation=tf.nn.relu,data_format='channels_last')


        upsample6 = tf.keras.layers.UpSampling3D(size=2,  data_format='channels_last')(conv12_3)
        # print(upsample3) 8x320x240x64
        
        concat8 = tf.concat([conv1, upsample6], axis = 4, name = 'concat8')

        conv13_1 = tf.layers.conv3d(concat8,32,3,padding='same',name="45",activation=tf.nn.relu,data_format='channels_last')  
        conv13_2 = tf.layers.conv3d(conv13_1,32,3,padding='same',name="46",activation=tf.nn.relu,data_format='channels_last') 
        conv13_3 = tf.layers.conv3d(conv13_2,32,3,padding='same',name="47",activation=tf.nn.relu,data_format='channels_last')

        output = tf.layers.conv3d(conv13_3,3,3,padding='same',name="48",activation=tf.nn.sigmoid,data_format='channels_last') 
        print(output)
        return output
 
# input_shape=(None,8,320,240,3)
# container = tf.placeholder(shape=input_shape, dtype=tf.float32, name='conatiner_input')
# revealing_network(container)