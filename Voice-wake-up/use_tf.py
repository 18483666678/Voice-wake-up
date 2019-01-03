import tensorflow as tf
import os
import numpy as np
from pydub import AudioSegment
import time
import pygame
from cnn2seq import Net

class  USE:

    def __init__(self):

        self.net=Net()

    def Use(self,data):
        saver = tf.train.Saver(max_to_keep=1)
        init=tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)
            if os.path.exists(r"C:\Users\liev\Desktop\log\语音\model\checkpoint"):
                new_saver = tf.train.latest_checkpoint(r"C:\Users\liev\Desktop\log\语音\model")
                saver.restore(sess, new_saver)



            #(19008,)to(1,19008,1)
                data_=np.reshape(data,(1,-1,1))

                output=sess.run(self.net.out,feed_dict={self.net.x:data_,self.net.istrain:False})

                return output


if __name__ == '__main__':
    use=USE()
    path=r"D:\PycharmProjects\语音唤醒\shengyin.mp3\小白小白.mp3"

    # input输入语音

    xiaobai=AudioSegment.from_mp3(path)
    x=xiaobai.get_array_of_samples()
    x_=np.array(x,dtype=np.float32)
    data=x_*(1.0)/(max(abs(x_)))

    while len(data)<49536:
        data=np.append(data,0)
    data=np.array(data,dtype=np.float32)

    out=use.Use(data)

    print(out)
    if out>0.9:
        print("hello world")
        pygame.mixer.init()
        track = pygame.mixer.music.load(r"D:\语音\播放\0-0.mp3")
        pygame.mixer.music.play()
        time.sleep(4)
        pygame.mixer.music.stop()
    else:
        print("呼叫错误")