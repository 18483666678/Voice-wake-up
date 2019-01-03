import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pydub import AudioSegment#读MP3
import time
import pygame#播放
from torch.autograd import Variable
from cnn2seq import Encoder,Decoder

class USE:

    def __init__(self):

        self.encoder=Encoder()
        self.decoder=Decoder()

        self.encoder=self.encoder.cuda()
        self.decoder=self.decoder.cuda()

        self.encoder.load_state_dict(torch.load(r"Param/encoder.pt"))
        self.decoder.load_state_dict(torch.load(r"Param/decoder.pt"))

        self.encoder.eval()
        self.decoder.eval()

    def Use(self,x):
        x=Variable(x)
        x=x.cuda()
        x.unsqueeze_(0)#升维(L,)to(N,L)
        x_ = x.view(1,-1,1).permute(0,2,1)#(N,C,L)
        self.encoder_out=self.encoder.forward(x)
        self.decoder_out=self.decoder.forward(self.encoder_out)#(N<C)

        return self.decoder_out

if __name__ == '__main__':
    use=USE()

    path=r"E:\shengyin\小白小白.mp3"
    data=AudioSegment.from_mp3(path)
    data_=data.get_array_of_samples()
    x_=np.array(data_,dtype=np.float32)
    _x_=x_*(1.0)/(max(abs(x_)))#归一化
    while len(_x_)<49536:
        _x_=np.append(_x_,0)
    x=np.array(_x_,dtype=np.float32)

    out=use.Use(x)

    if out>0.9:
        print("hello world")
        pygame.mixer.init()#初始化音频模块
        track = pygame.mixer.music.load(r"D:\语音\播放\0-0.mp3")#加载音乐
        pygame.mixer.music.play()#        播放载入的音乐
        time.sleep(4)#播放完等4秒
        pygame.mixer.music.stop()#        停止播放
    else:
        print("呼叫错误")


