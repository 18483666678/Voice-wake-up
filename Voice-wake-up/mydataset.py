from pydub import AudioSegment
import numpy as np
import tensorflow as tf
import os

#打开音频
# sound2 =AudioSegment.from_mp3(r"D:\语音\音频\0-0.mp3")
# print(sound2)
# a=sound2.get_array_of_samples()
# a_=np.array(a,np.float32)
# print(a_)
# b=r"0-0.mp3"
# c=b.split("-")
# print(c[0])

class Mydataset:

    def __init__(self,path):
        self.path=path

        self.files=[]
        for file in os.listdir(self.path):
            self.files.append(file)
        np.random.shuffle(self.files)

        self.dataset=tf.data.Dataset.from_tensor_slices(self.files)
        self.dataset=self.dataset.map(lambda file:tuple(tf.py_func(self.read_data,[file],[tf.float32,tf.int32])))
        self.dataset=self.dataset.shuffle(200)
        self.dataset=self.dataset.repeat()
        self.dataset=self.dataset.batch(50)

        self.iterator=self.dataset.make_one_shot_iterator()
        self.item=self.iterator.get_next()

    def read_data(self,file):
        file=file.decode()
        data_path=os.path.join(self.path,file)
        data=AudioSegment.from_mp3(data_path)
        data_=data.get_array_of_samples()
        x_=np.array(data_,dtype=np.float32)
        x=x_*1.0/(max(abs(x_)))
        num=49536
        while len(x)<num:
            x=np.append(x,0)

        x=np.array(x,dtype=np.float32)


        y=int(file.split("-")[0])



        return x,y

    def get_batch(self,sess):
        return sess.run(self.item)

if __name__ == '__main__':
    my=Mydataset(r"D:\语音\音频")
    with tf.Session() as sess:

        xs,ys=my.get_batch(sess)
        print(xs.shape)
        print(ys.shape)
        # for x in xs:
        #     print(x.shape)