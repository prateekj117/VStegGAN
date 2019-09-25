# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 2019

@author: Suraj, am7
"""

import hide_net1
import reveal_net1
import tensorflow as tf
import os
import cv2
import numpy as np
import pickle
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

test_loc = '../../Suraj2/ucf/test/'
train_loc = '../../Suraj2/ucf/train/'
input_shape_cover = (None, 8, 240, 320, 3)
input_shape_secret = (None, 8, 240, 320, 3)
beta = 0.75
EPS=1e-12

def convert(x):
    return int(x)


class SingleSizeModel():
    # def get_noise_layer_op(self,tensor,std=.1):
    #     with tf.variable_scope("noise_layer"):
    #         return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32)

    def __init__(self, input_shape_cover, input_shape_secret, beta, EPS):

        self.checkpoint_dir = 'checkpoints_s'
        self.model_name = 'VStegNet'
        self.dataset_name = 'ucf'
        self.test_dir_all = 'test_s'
        self.log_dir = 'logs_s'
        self.img_height = 240
        self.img_width = 320
        self.channels = 3
        self.frames_per_batch = 8
        self.batch_size = 1

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.test_dir_all):
            os.makedirs(self.test_dir_all)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.cover_tensor_data_test, self.secret_tensor_data_test = self.get_test_data()
        self.tensor_data_train = self.get_train_data()

        self.beta = beta
        self.eps = EPS
        self.learning_rate = 0.0001
        self.sess = tf.InteractiveSession()

        self.secret_tensor = tf.placeholder(shape=input_shape_secret, dtype=tf.float32, name="input_secret")
        self.cover_tensor = tf.placeholder(shape=input_shape_cover, dtype=tf.float32, name="input_cover")
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        self.train_op_cov, self.minimize_gen, self.minimize_dis, self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.gen_loss, self.dis_loss = self.prepare_training_graph(self.secret_tensor,self.cover_tensor,self.global_step_tensor)
        #self.hiding_output, self.reveal_output, self.summary_op, self.loss_op, self.secret_loss_op, self.cover_loss_op, self.gen_loss, self.dis_loss= self.prepare_test_graph(self.cover_tensor, self.secret_tensor)

        self.writer = tf.summary.FileWriter(self.log_dir,self.sess.graph)

        self.sess.run(tf.global_variables_initializer())


    def get_test_data(self):

        dirs = [test_loc + dir_name for dir_name in os.listdir(test_loc)]
        dirs.extend([train_loc + dir_name for dir_name in os.listdir(train_loc)])
        indices = np.random.choice(len(dirs), 500)
        covers = []
        secrets = []
        for i in range(len(indices)):
            covers.append(dirs[indices[i]])
            index = np.random.choice(len(dirs))
            while dirs[index] == covers[i]:
                index = np.random.choice(len(dirs))
            secrets.append(dirs[index])

        return covers, secrets


    def get_train_data(self):

        dirs = os.listdir(train_loc)
        if 'train' in dirs:
        	dirs.remove('train')
        dirs = sorted(dirs, key=convert)
        dat = []
        for i in range(len(dirs)):
            dat.append(train_loc + dirs[i])

        return dat


    def prepare_training_graph(self,secret_tensor,cover_tensor,global_step_tensor):

        hiding_output = hide_net1.hiding_network(cover_tensor, secret_tensor)
        reveal_output = reveal_net1.revealing_network(hiding_output)

        with tf.name_scope("discriminato"):
            with tf.variable_scope('d_net'):
                p_real = discriminator_net1.discriminating_network(cover_tensor)
            with tf.variable_scope('d_net', reuse=True):
                p_fake = discriminator_net1.discriminating_network(hiding_output)

        loss_op, secret_loss_op, cover_loss_op, gen_loss, dis_loss = self.get_loss_op(secret_tensor, reveal_output, cover_tensor, hiding_output, p_real, p_fake, beta=self.beta, EPS=self.eps)

        t_vars = tf.trainable_variables()
        h_vars = [var for var in t_vars if 'h_net' in var.name]
        d_vars = [var for var in t_vars if 'd_net' in var.name]
        # r_vars = [var for var in t_vars if 'reveal_net' in var.name]

        minimize_op_cov = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op, global_step=global_step_tensor)
        minimize_gen = tf.train.AdamOptimizer(0.00001).minimize(gen_loss, var_list=h_vars, global_step=global_step_tensor)
        minimize_dis = tf.train.AdamOptimizer(0.00001).minimize(dis_loss, var_list=d_vars, global_step=global_step_tensor)

        tf.summary.scalar('loss', loss_op, family='train')
        tf.summary.scalar('reveal_net_loss', secret_loss_op, family='train')
        tf.summary.scalar('cover_net_loss', cover_loss_op, family='train')
        tf.summary.scalar('generator_loss', gen_loss, family='train')
        tf.summary.scalar('discriminator_loss', dis_loss, family='train')

        merged_summary_op = tf.summary.merge_all()

        return minimize_op_cov, minimize_gen, minimize_dis, merged_summary_op, loss_op, secret_loss_op, cover_loss_op, gen_loss, dis_loss


    def prepare_test_graph(self, cover_tensor, secret_tensor):
        with tf.variable_scope("",reuse=True):

            hiding_output = hide_net1.hiding_network(cover_tensor, secret_tensor)
            reveal_output = reveal_net1.revealing_network(hiding_output)

            p_real = discriminator_net1.discriminating_network(cover_tensor)
            p_fake = discriminator_net1.discriminating_network(hiding_output)

            loss_op, secret_loss_op, cover_loss_op, gen_loss, dis_loss = self.get_loss_op(secret_tensor, reveal_output, cover_tensor, hiding_output, p_real, p_fake)

            tf.summary.scalar('loss', loss_op,family='test')
            tf.summary.scalar('reveal_net_loss', secret_loss_op,family='test')
            tf.summary.scalar('cover_net_loss', cover_loss_op,family='test')
            tf.summary.scalar('generator_loss', gen_loss, family='test')
            tf.summary.scalar('discriminator_loss', dis_loss, family='test')

            merged_summary_op = tf.summary.merge_all()

            return hiding_output, reveal_output, merged_summary_op, loss_op, secret_loss_op, cover_loss_op, gen_loss, dis_loss


    def get_loss_op(self, secret_true, secret_pred, cover_true, cover_pred, p_real, p_fake, beta=0.75, EPS=1e-12):

        with tf.variable_scope("losses"):

            beta = tf.constant(beta, name="beta")
            secret_mse = tf.losses.mean_squared_error(secret_true, secret_pred)
            cover_mse = tf.losses.mean_squared_error(cover_true, cover_pred)

            gen_loss = tf.reduce_mean(-tf.log(p_fake+EPS))
            dis_loss = tf.reduce_mean(-(tf.log(p_real+EPS) + tf.log(1-p_fake+EPS)))

            final_loss = cover_mse + beta * secret_mse

            return final_loss , secret_mse , cover_mse, gen_loss, dis_loss


    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.model_name, self.dataset_name, self.img_height, self.img_width, self.beta)

    @property
    def test_dir(self):
        return "{}_{}_{}_{}_{}_test".format(
            self.model_name, self.dataset_name, self.img_height, self.img_width, self.beta)


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'stegnet.model-5038'))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


    def train(self):

        vids = 100000

        self.saver = tf.train.Saver(max_to_keep=2)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            count = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            count = 0
            print(" [!] Load failed...")


        def load_t(base_dir, frame_names, ind):

            batch = []
            for i in range(self.frames_per_batch):
                frame = base_dir + '/' + frame_names[ind * self.frames_per_batch + i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((self.img_height, self.img_width, self.channels))
                batch.append(frame)

            return np.array(batch)

        def generator():

            training_batch_covers = []
            training_batch_secrets = []
            b = 0

            for i in range(count, vids):
                c_path = np.random.choice(self.tensor_data_train)
                s_path = np.random.choice(self.tensor_data_train)
                while s_path == c_path:
                    s_path = np.random.choice(self.tensor_data_train)
                n1 = len(os.listdir(c_path))
                n2 = len(os.listdir(s_path))
                t = int(min(n1, n2) / self.frames_per_batch)
                frs = sorted(os.listdir(c_path), key=lambda x: int(x.split('.')[0]))[:t * self.frames_per_batch]
                for j in range(t):
                    cov_tens = load_t(c_path, frs, j)
                    sec_tens = load_t(s_path, frs, j)
                    b += 1
                    training_batch_covers.append(cov_tens)
                    training_batch_secrets.append(sec_tens)
                    if b == self.batch_size:
                        yield np.array(training_batch_covers), np.array(training_batch_secrets), i
                        b = 0
                        training_batch_covers = []
                        training_batch_secrets = []

            if b:
                yield np.array(training_batch_covers), np.array(training_batch_secrets), i


        print ('Beginning training ...')
        start_time = time()

        i = 1
        for covers, secrets, vid in generator():

            _, _, _, gs, tl, sl, cl, gl, dl = self.sess.run([self.train_op_cov, self.minimize_gen, self.minimize_dis, self.global_step_tensor,
                                    self.loss_op, self.secret_loss_op, self.cover_loss_op, self.gen_loss, self.dis_loss],
                                    feed_dict={"input_secret:0":secrets, "input_cover:0":covers})

            if i % 10 == 0:
                summaree, gs = self.sess.run([self.summary_op, self.global_step_tensor],
                                                feed_dict={"input_secret:0":secrets,"input_cover:0":covers})
                self.writer.add_summary(summaree, gs)

            i += 1

            print('Video: '+str(vid)+' Iteration: '+str(i)+' Time: '+str(time() - start_time)
                    +' Loss: '+str(tl)+' Cover_Loss: '+str(cl)+' Secret_Loss: '+str(sl)+' Gen_Loss: '+str(gl)+' Dis_Loss '+str(dl))

            if i % 200 == 0:
                self.save(self.checkpoint_dir, vid)


    def test(self):

        self.saver = tf.train.Saver(max_to_keep=2)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit()

        def load_t(base_dir, frame_names, ind):

            batch = []
            for i in range(self.frames_per_batch):
                frame = base_dir + '/' + frame_names[ind * self.frames_per_batch + i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((self.img_height, self.img_width, self.channels))
                batch.append(frame)

            return np.array([batch])

        def generator():
            for i in range(len(self.cover_tensor_data_test)):
                c_name = self.cover_tensor_data_test[i].split('/')[-1]
                s_name = self.secret_tensor_data_test[i].split('/')[-1]
                n1 = len(os.listdir(self.cover_tensor_data_test[i]))
                n2 = len(os.listdir(self.secret_tensor_data_test[i]))
                t = int(min(n1, n2) / self.frames_per_batch)
                frs = sorted(os.listdir(self.cover_tensor_data_test[i]), key=lambda x: int(x.split('.')[0]))[:t * self.frames_per_batch]
                for j in range(t):
                    cov_tens = load_t(self.cover_tensor_data_test[i], frs, 0)
                    sec_tens = load_t(self.secret_tensor_data_test[i], frs, 0)
                    yield cov_tens, sec_tens, c_name, s_name, i

        def load_random(base_dir, frame_names):
            batch = []
            for i in range(self.frames_per_batch):
                frame = base_dir + '/' + frame_names[i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((self.img_height, self.img_width, self.channels))
                batch.append(frame)

            return np.array([batch])

        def generator_random():
            for i in range(5):
                p = np.random.choice(len(self.cover_tensor_data_test))
                c_name = self.cover_tensor_data_test[p].split('/')[-1]
                s_name = self.secret_tensor_data_test[p].split('/')[-1]
                frame_names = sorted(os.listdir(self.cover_tensor_data_test[p]), key=lambda x: int(x.split('.')[0]))
                frame_names2 = os.listdir(self.secret_tensor_data_test[p])
                p1 = np.random.choice(min(len(frame_names), len(frame_names2)), self.frames_per_batch)
                frame_names = np.array(frame_names)[p1]
                cov_tens = load_random(self.cover_tensor_data_test[p], frame_names)
                sec_tens = load_random(self.secret_tensor_data_test[p], frame_names)
                yield cov_tens, sec_tens, c_name, s_name, i

        def load_random_all(frame_names):
            batch = []
            for i in range(self.frames_per_batch):
                frame = frame_names[i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((self.img_height, self.img_width, self.channels))
                batch.append(frame)

            return np.array([batch])

        def generator_random_all():
            # for i in range(len(self.cover_tensor_data_test)):
            for j in range(5):
                p = np.random.choice(len(self.cover_tensor_data_test), self.frames_per_batch)
                c_name = ''
                for i in range(self.frames_per_batch):
                    c_name += self.cover_tensor_data_test[p[i]].split('/')[-1]
                    if i < self.frames_per_batch - 1:
                        c_name += '_'
                s_name = ''
                for i in range(self.frames_per_batch):
                    s_name += self.secret_tensor_data_test[p[i]].split('/')[-1]
                    if i < self.frames_per_batch - 1:
                        s_name += '_'
                p1 = np.random.choice(9)
                frame_names = []
                file_names = []
                for i in range(self.frames_per_batch):
                    frame_names.append(self.cover_tensor_data_test[p[i]] + '/00' + str(p1 + 1) + '.jpg')
                    file_names.append(str(i)+'_00' + str(p1 + 1) + '.jpg')
                cov_tens = load_random_all(frame_names)
                frame_names = []
                for i in range(self.frames_per_batch):
                    frame_names.append(self.secret_tensor_data_test[p[i]] + '/00' + str(p1 + 1) + '.jpg')
                sec_tens = load_random_all(frame_names)
                yield cov_tens, sec_tens, c_name, s_name, file_names, j


        print ('Beginning testing ...')


        start_time = time()
        total_frames = 0
        i = 0
        prev_vid = 0
        # for test_cover, test_secret, c_name, s_name, file_names, vid in generator_random_all():
        for test_cover, test_secret, c_name, s_name, vid in generator():

            if prev_vid != vid:
                i = 0
                prev_vid = vid

            print ('Testing video: ' + str(vid) + ', c_name: '+c_name+', s_name: '+s_name)

            cover_loss = 0
            secret_loss = 0
            cover_accuracy = 0

            test_dir = os.path.join(self.test_dir_all, self.test_dir)
            video_dir = os.path.join(test_dir, 'c_'+c_name+'_s_'+s_name)
            cover_dir = os.path.join(video_dir, 'cover')
            container_dir = os.path.join(video_dir, 'container')
            secret_dir = os.path.join(video_dir, 'secret')
            revealed_secret_dir = os.path.join(video_dir, 'revealed_secret')
            diff_cover_container_dir = os.path.join(video_dir, 'diff_cc')
            diff_secret_revealed_dir = os.path.join(video_dir, 'diff_sr')

            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
                os.makedirs(cover_dir)
                os.makedirs(container_dir)
                os.makedirs(secret_dir)
                os.makedirs(revealed_secret_dir)
                os.makedirs(diff_cover_container_dir)
                os.makedirs(diff_secret_revealed_dir)

            # hiding_b, reveal_b, c11, c12, c13, c14, c1, c21, c22, c23, c2, c3, c4, c5, c6, c7, c8, c9, c11_r, c12_r, c13_r, c14_r, c1_r, c21_r, c22_r, c23_r, c2_r, c3_r, c4_r, c5_r, c6_r, c7_r, c8_r, c9_r = self.sess.run([
            #     self.hiding_output, self.reveal_output, self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4, self.conv1, self.conv2_1, self.conv2_2, self.conv2_3, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.conv9,
            #     self.conv1_1_r, self.conv1_2_r, self.conv1_3_r, self.conv1_4_r, self.conv1_r, self.conv2_1_r, self.conv2_2_r, self.conv2_3_r, self.conv2_r, self.conv3_r, self.conv4_r, self.conv5_r, self.conv6_r, self.conv7_r, self.conv8_r, self.conv9_r],
            #                                     feed_dict={"input_secret:0": test_secret, "input_cover:0": test_cover})
            hiding_b, reveal_b = self.sess.run([self.hiding_output, self.reveal_output],
                                                feed_dict={"input_secret:0": test_secret, "input_cover:0": test_cover})

            # for j in range(self.frames_per_batch):
            #     im = np.reshape(hiding_b[0][j] * 255, (self.img_height, self.img_width, self.channels))
            #     cv2.imwrite(container_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)

            #     im = np.reshape(reveal_b[0][j] * 255, (self.img_height, self.img_width, self.channels))
            #     cv2.imwrite(revealed_secret_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)

            #     im = np.reshape(test_cover[0][j] * 255, (self.img_height, self.img_width, self.channels))
            #     cv2.imwrite(cover_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)

            #     im = np.reshape(test_secret[0][j] * 255, (self.img_height, self.img_width, self.channels))
            #     cv2.imwrite(secret_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)

            #     im = np.reshape(np.absolute(hiding_b[0][j] - test_cover[0][j]) * 255, (self.img_height, self.img_width, self.channels))
            #     cv2.imwrite(diff_cover_container_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)

            #     im = np.reshape(np.absolute(reveal_b[0][j] - test_secret[0][j]) * 255, (self.img_height, self.img_width, self.channels))
            #     cv2.imwrite(diff_secret_revealed_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)

            # np.save(video_dir+'/c11', c11[0])
            # np.save(video_dir+'/c12', c12[0])
            # np.save(video_dir+'/c13', c13[0])
            # np.save(video_dir+'/c14', c14[0])
            # np.save(video_dir+'/c1', c1[0])
            # np.save(video_dir+'/c21', c21[0])
            # np.save(video_dir+'/c22', c22[0])
            # np.save(video_dir+'/c23', c23[0])
            # np.save(video_dir+'/c2', c2[0])
            # np.save(video_dir+'/c3', c3[0])
            # np.save(video_dir+'/c4', c4[0])
            # np.save(video_dir+'/c5', c5[0])
            # np.save(video_dir+'/c6', c6[0])
            # np.save(video_dir+'/c7', c7[0])
            # np.save(video_dir+'/c8', c8[0])
            # np.save(video_dir+'/c9', c9[0])
            # np.save(video_dir+'/c11_r', c11_r[0])
            # np.save(video_dir+'/c12_r', c12_r[0])
            # np.save(video_dir+'/c13_r', c13_r[0])
            # np.save(video_dir+'/c14_r', c14_r[0])
            # np.save(video_dir+'/c1_r', c1_r[0])
            # np.save(video_dir+'/c21_r', c21_r[0])
            # np.save(video_dir+'/c22_r', c22_r[0])
            # np.save(video_dir+'/c23_r', c23_r[0])
            # np.save(video_dir+'/c2_r', c2_r[0])
            # np.save(video_dir+'/c3_r', c3_r[0])
            # np.save(video_dir+'/c4_r', c4_r[0])
            # np.save(video_dir+'/c5_r', c5_r[0])
            # np.save(video_dir+'/c6_r', c6_r[0])
            # np.save(video_dir+'/c7_r', c7_r[0])
            # np.save(video_dir+'/c8_r', c8_r[0])
            # np.save(video_dir+'/c9_r', c9_r[0])


            # total_frames += self.frames_per_batch

            for j in range(self.frames_per_batch):
                im = np.reshape(hiding_b[0][j] * 255, (self.img_height, self.img_width, self.channels))
                cv2.imwrite(container_dir+'/'+file_names[j], im)

                im = np.reshape(reveal_b[0][j] * 255, (self.img_height, self.img_width, self.channels))
                cv2.imwrite(revealed_secret_dir+'/'+file_names[j], im)

                im = np.reshape(test_cover[0][j] * 255, (self.img_height, self.img_width, self.channels))
                cv2.imwrite(cover_dir+'/'+file_names[j], im)

                im = np.reshape(test_secret[0][j] * 255, (self.img_height, self.img_width, self.channels))
                cv2.imwrite(secret_dir+'/'+file_names[j], im)

                im = np.reshape(np.absolute(hiding_b[0][j] - test_cover[0][j]) * 255, (self.img_height, self.img_width, self.channels))
                cv2.imwrite(diff_cover_container_dir+'/'+file_names[j], im)

                im = np.reshape(np.absolute(reveal_b[0][j] - test_secret[0][j]) * 255, (self.img_height, self.img_width, self.channels))
                cv2.imwrite(diff_secret_revealed_dir+'/'+file_names[j], im)

            total_frames += self.frames_per_batch

            i += 1

        total_time = time() - start_time
        pickle.dump(total_time, open('total_time_new.pkl', 'wb'))
        time_per_frame = float(total_time) / total_frames
        pickle.dump(time_per_frame, open('time_per_frame_new.pkl', 'wb'))
        print ('Total time: '+str(total_time)+' Time Per Frame: '+str(time_per_frame))


def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)


m = SingleSizeModel(input_shape_cover=input_shape_cover, input_shape_secret=input_shape_secret, beta=beta, EPS=EPS)
# show_all_variables()
m.train()
# m.test()
