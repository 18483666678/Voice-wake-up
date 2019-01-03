import tensorflow as tf
import numpy as np
import os
import wave
from pydub import AudioSegment
import pygame
import time
from mydataset import Mydataset


class Encoder:

    def forward(self, x, istrain):
        with tf.variable_scope("encoder"):
            # (N,49536,1)TO(N,1546,18)
            self.conv1 = tf.nn.leaky_relu(tf.layers.conv1d(x, 18, kernel_size=64, strides=32, padding="valid"))
            # (N,1546,18)TO(N,94,36)
            self.conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(
                tf.layers.conv1d(self.conv1, 36, kernel_size=32, strides=16, padding="valid"), training=istrain))
            # (N,94,36)to(N,9,72)
            self.conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(
                tf.layers.conv1d(self.conv2, 72, kernel_size=16, strides=8, padding="valid"), training=istrain))
            # (N,9,72)to(N,4,128)
            self.conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(
                tf.layers.conv1d(self.conv3, 128, kernel_size=3, strides=2, padding="valid"), training=istrain))

            return self.conv4


class Decoder:

    def __init__(self):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(128)
        self.mlstm = tf.nn.rnn_cell.MultiRNNCell([self.cell] * 3)

        # 训练
        # self.init_state=self.mlstm.zero_state(50,dtype=tf.float32)
        # 使用
        self.init_state = self.mlstm.zero_state(1, dtype=tf.float32)

    def forward(self, x):
        self.decoder_outs, self.finally_state = tf.nn.dynamic_rnn(self.mlstm, x, initial_state=self.init_state,
                                                                  time_major=False)

        return self.decoder_outs


class Net:

    def __init__(self):
        # (N,L,C)
        self.x = tf.placeholder(shape=(None, None, 1), dtype=tf.float32)
        self.y = tf.placeholder(shape=(None, 1), dtype=tf.float32)

        self.istrain = tf.placeholder(tf.bool)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.forwar()
        self.backward()

    def forwar(self):
        self.encoder_out = self.encoder.forward(self.x, self.istrain)
        self.decoder_out = self.decoder.forward(self.encoder_out)

        self.finally_flat = self.decoder_out[:, -1, :]

        self.out = tf.nn.sigmoid(tf.layers.dense(self.finally_flat, 1))

    def backward(self):
        self.loss = tf.reduce_mean((self.out - self.y) ** 2)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    net = Net()
    sample = Mydataset(r'D:\语音\音频')
    saver = tf.train.Saver(max_to_keep=1)
    model_path = r"model/model.ckpt"
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        if os.path.exists(r"C:\Users\liev\Desktop\log\语音\model\checkpoint"):
            new_saver = tf.train.latest_checkpoint(r"C:\Users\liev\Desktop\log\语音\model")
            saver.restore(sess, new_saver)

        # for epoch in range(10000):
        #
        #     xs,ys=sample.get_batch(sess)
        #     xs=np.reshape(xs,(50,-1,1))
        #     ys=np.reshape(ys,(50,1))
        #
        #     out,loss,opt=sess.run([net.out,net.loss,net.opt],feed_dict={net.x:xs,net.y:ys,net.istrain:True})
        #     print(epoch,"次：","loss",loss)
        #     print("ys",ys[:5],"out",out[:5])
        #
        #     if epoch%10==0:
        #         saver.save(sess,model_path)
        #         print("保存")

        # 使用
        path = r"D:\语音\小白大亮\0-1.mp3"

        # input输入语音

        xiaobai = AudioSegment.from_mp3(path)
        x = xiaobai.get_array_of_samples()
        x_ = np.array(x, dtype=np.float32)
        data = x_ * (1.0) / (max(abs(x_)))

        while len(data) < 49536:
            data = np.append(data, 0)
        data_ = np.array(data, dtype=np.float32)
        Data = np.reshape(data_, (1, -1, 1))

        out = sess.run(net.out, feed_dict={net.x: Data, net.istrain: False})
        print(out)
        if out > 0.9:
            print("hello world")
            pygame.mixer.init()
            track = pygame.mixer.music.load(r"D:\语音\播放\0-0.mp3")
            pygame.mixer.music.play()
            time.sleep(4)
            pygame.mixer.music.stop()
        else:
            print("呼叫错误")
