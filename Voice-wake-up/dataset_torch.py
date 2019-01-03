import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from pydub import AudioSegment
from torch.autograd import Variable

class Sample(Dataset):
    def __init__(self,path):
        self.path=path
        self.data=[]

        for file in os.listdir(self.path):
            self.data.append(file)
        np.random.shuffle(self.data)


    def __getitem__(self,item):
        strs_path=os.path.join(self.path,self.data[item])
        data_mp3=AudioSegment.from_mp3(strs_path)
        data_mp3_=data_mp3.get_array_of_samples()
        Data=np.array(data_mp3_,dtype=np.float32)
        Data_=Data*(1.0)/(max(abs(Data)))
        num=49536
        while len(Data_)<num:
            Data_=np.append(Data_,0)
        x=np.array(Data_,dtype=np.float32)
        y=int(self.data[item].split("-")[0])

        return x,y

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    my=Sample(r"D:\语音\音频")
    dataloader=DataLoader(my,batch_size=5,shuffle=True,num_workers=3)

    for i,(x,y) in enumerate(dataloader):
        x=Variable(x)
        y=Variable(y)

        x=x.cuda()
        y=y.cuda()

        print(x)
        print(y)
        print(x.shape)
        print(y.shape)