import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
import wave

class Mydataset(Dataset):
    def __init__(self,path):
        self.path=path

        self.dataset=[]
        for file in os.listdir(os.path.join(self.path,r"positive")):
            self.dataset.append(os.path.join(self.path,r"positive",file))
        for file in os.listdir(os.path.join(self.path,r"negative")):
            self.dataset.append(os.path.join(self.path,r"negative",file))

        np.random.shuffle(self.dataset)

    def __getitem__(self,item):

        data_path=self.dataset[item]
        data=wave.open(data_path,"rb")
        parm=data.getparams()

        #nchannels: 声道数,sampwidth: 量化位数（byte）,framerate: 采样频率,nframes: 采样点数
        nchannels,sampwidth,framerate,nframes=parm[:4]
        strdata=data.readframes(nframes)
        wavedata=np.fromstring(strdata,dtype=np.float32)
        x=wavedata*1.0/(max(abs(wavedata)))

        if os.path.samefile(data_path,os.path.join(self.path,r"positive",data_path.split("\\")[-1])):
            y=1
        elif os.path.samefile(data_path,os.path.join(self.path,r"negative",data_path.split("\\")[-1])):
            y=0

        return x,y

    def __len__(self):
        return len(self.dataset)
