import tensorflow as tf
import wave
import os
import numpy as np


# class Sample:
#
#     def __init__(self,path):
#
#         self.path=path
#         files=[]
#         ys=[]
#         for file in os.listdir(self.path):
#             files.append(os.path.join(self.path,file))
#             ys.append(int(file.split(".")[0]))
#         self.files=files
#         self.ys=ys
#
#
#         self.dataset=tf.data.Dataset.from_tensor_slices((self.files,self.ys))
#         self.dataset=self.dataset.map(lambda file,y:tuple(tf.py_func(self.read_data,[file,y],[tf.float32,y.dtype])))
#
#         self.dataset=self.dataset.shuffle(2)
#         self.dataset=self.dataset.repeat()
#         self.dataset=self.dataset.batch(1)
#
#         self.iterator=self.dataset.make_one_shot_iterator()
#         self.item=self.iterator.get_next()
#
#     def read_data(self,file,y):
#
#         f=wave.open(file.decode(),"rb")
#         parm=f.getparams()
#         nchannel,sampwidth,framerate,nframes=parm[:4]
#         strdata=f.readframes(nframes)
#         wavedata=np.fromstring(strdata,dtype=np.float32)
#         x=wavedata*1.0/(max(abs(wavedata)))
#
#         print(x,"111111111111")
#         print(y,"222222222222222222222")
#         return x,y
#
#     def get_batch(self,sess):
#         return sess.run(self.item)
#
# if __name__ == '__main__':
#     sample = Sample(r"C:\Users\liev\Desktop\log\语音\音频")
#
#     with tf.Session() as sess:
#         xs,ys=sample.get_batch(sess)
#         print(xs.shape,"333333333333333333333")
#         print(ys,"44444444444444444444444444")


# a=1
# if a==1:
#     print("1")
#     a=0
# elif a==0:
#     print("0")


import os
# a=[[1,2,3],[2,3,4]]
# b=[1,0]
# file=os.path.join(r"C:\Users\liev\Desktop\音频","positive","1.wav")
# print(file.split("\\")[-1])
# f=os.path.join(r"C:\Users\liev\Desktop\音频","positive",file.split("\\")[-1])
# print(f)
# if os.path.samefile(file,os.path.join(r"C:\Users\liev\Desktop\音频","positive",file.split("\\")[-1])):
#     y=1
# print(y)

# class Sample:
#
#     def __init__(self,path):
#
#         self.path=path
#         files=[]
#         for file in os.listdir(os.path.join(self.path,r"positive")):
#             files.append(os.path.join(self.path,r"positive",file))
#
#         for file in os.listdir(os.path.join(self.path,"negative")):
#             files.append(os.path.join(self.path, r"negative", file))
#         self.files=files
#
#         np.random.shuffle(self.files)
#
#         self.dataset=tf.data.Dataset.from_tensor_slices(self.files)
#         self.dataset=self.dataset.map(lambda file:tuple(tf.py_func(self.read_data,[file],[tf.float32,tf.int32])))
#
#         self.dataset=self.dataset.shuffle(2)
#         self.dataset=self.dataset.repeat()
#         self.dataset=self.dataset.batch(1)
#
#         self.iterator=self.dataset.make_one_shot_iterator()
#         self.item=self.iterator.get_next()
#
#     def read_data(self,file):
#         file=file.decode()
#         f=wave.open(file,"rb")
#         parm=f.getparams()
#         nchannel,sampwidth,framerate,nframes=parm[:4]
#         strdata=f.readframes(nframes)
#         wavedata=np.fromstring(strdata,dtype=np.float32)
#         x=wavedata*1.0/(max(abs(wavedata)))
#
#         if os.path.samefile(file,os.path.join(self.path,r"positive",file.split("\\")[-1])):
#             y=1
#         elif os.path.samefile(file,os.path.join(self.path,r"negative",file.split("\\")[-1])):
#             y=0
#
#         print(x,"11111111111111")
#         print(y,"2222222222222222222")
#
#         return x,y
#
#     def get_batch(self,sess):
#         return sess.run(self.item)
#
# if __name__ == '__main__':
#     sample = Sample(r"C:\Users\liev\Desktop\音频")
#
#     with tf.Session() as sess:
#         xs,ys=sample.get_batch(sess)
#         print(xs.shape,"333333333333333333333")
#         print(ys,"44444444444444444444444444")


# a=np.array([1,23])
# b=np.array([11,22])
# c=[]
# c.append(a)
# c.append(b)
# for i in c:
#     print(i)
#     np.savez("1.npz",i)
#
# d=np.load("1.npz")
# for j in d:
#     print(d[j])
# strs=open(r"D:\语音\标签\yangben.txt").readlines()
# c=0
# for data in strs:
#     data=data.strip()
#     print(data)
#     print(c)
#     c+=1
from pydub import AudioSegment
import numpy as np
import tensorflow as tf
import os
# path=r'D:\语音\音频'
# sh=[]
# for file in os.listdir(path):
#     data_path = os.path.join(path, file)
#     data = AudioSegment.from_mp3(data_path)
#     data_ = data.get_array_of_samples()
#     x_ = np.array(data_, dtype=np.float32)
#     x = x_ * 1.0 / (max(abs(x_)))
#     print(len(x),"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#     print(x,"#########################################")
#     while len(x)<49536:
#         x=np.append(x,0)
#     print(x.shape[0],"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#     print(x,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#
#
#     # print(max(x.shape[0]))
#     # print(x.shape)
#     sh.append(np.max(x.shape[0]))
# #     print(x.shape,"++++++++++++++++++")
# #     print(np.max(x.shape[0]))
# # print(np.max(sh),"-------------")

# import tensorflow as tf
#
# import numpy as np
#
# a=np.array([[1],[0],[1],[0]])
# b=np.array([[0],[0],[1],[0]])
# acc=tf.reduce_mean(tf.cast(tf.equal(a,b),dtype=tf.float32))
#
# with tf.Session() as sess:
#     print(sess.run(acc))

data = AudioSegment.from_mp3(r"D:\语音\小白小白\0-0.mp3")
data_ = data.get_array_of_samples()
x_ = np.array(data_, dtype=np.float32)
x = x_ * 1.0 / (max(abs(x_)))
print(len(x),"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(x,"#########################################")