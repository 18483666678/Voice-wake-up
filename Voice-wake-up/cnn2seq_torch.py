import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from dataset_torch import Sample

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.conv=nn.Sequential(
            # (N,1,49536)to(N,18,1546)
            nn.Conv1d(1,18,kernel_size=64,stride=32,padding=0),
            nn.PReLU(),
            #(N,18,1546)to(N,94,36)
            nn.Conv1d(18,36,kernel_size=32,stride=16,padding=0),
            nn.BatchNorm1d(36),
            nn.PReLU(),
            #(N,94,36)to(N,72,9)
            nn.Conv1d(36,72,kernel_size=16,stride=8,padding=0),
            nn.BatchNorm1d(72),
            nn.PReLU(),
            #(N,72,9)to(N,128,4)
            nn.Conv1d(72,128,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(128),
            nn.PReLU()
        )

    def forward(self,x):
        #(N,c,L)
        self.y0=self.conv(x)

        return self.y0

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder,self).__init__()

        # 词向量维数,隐藏元维度,几个LSTM层串联,是否形状输入输出都是（N.L.C）      (128,128,3)
        self.lstm=nn.LSTM(input_size=128,hidden_size=128,num_layers=3,batch_first=True)
        self.linear=nn.Linear(128,1)

    def forward(self,x):
        #(N,l,C)c是输入维度
        x=x.permute(0,2,1)
        self.outs,self.hn=self.lstm(x)
        #（N,L,C）c是隐层个数
        self.out=self.outs[:,-1,:]
        #(N,c)
        self.output=F.sigmoid(self.linear(self.out))

        return self.output



if __name__ == '__main__':

    encoder = Encoder()
    decoder = Decoder()

    encoder = encoder.cuda()
    decoder = decoder.cuda()

    loss_fn=nn.BCELoss()
    encoder_opt=optim.Adam(encoder.parameters())
    decoder_opt=optim.Adam(decoder.parameters())

    sample=Sample(r"D:\语音\音频")
    dataloader=DataLoader(sample,batch_size=50,shuffle=True,num_workers=3)

    if os.path.exists(r"Param/encoder.pt"):
        encoder.load_state_dict(torch.load(r"Param/encoder.pt"))

    if os.path.exists(r"Param/decoder.pt"):
        decoder.load_state_dict(torch.load(r"Param/decoder.pt"))

    for i,(x,y) in enumerate(dataloader):
        x_=Variable(x)
        y_=Variable(y)

        x_=x_.cuda()
        y_=y_.cuda()

        _X_=x_.view(50,-1,1).permute(0,2,1)

        _Y_=y_.view(50,1)

        print(_X_.dtype,"111111111111111111111111")
        print(_X_.shape,"!!!!!!!!!!!!!!!!!!!")
        print(_Y_.dtype,"222222222222222222222222222")
        print(_Y_.shape,"@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        encoder_out = encoder.forward(_X_)

        print(encoder_out.shape,"333333333333333333")

        decoder_out = decoder.forward(encoder_out)

        print(decoder_out.shape,"4444444444444444444444444")

        loss=loss_fn(decoder_out,_Y_.float())

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()

        print(i,"次：","loss:",loss)
        print("y:",_Y_[:5],"out:",decoder_out[:5])

        if i%10==0:
            torch.save(encoder.state_dict(),r"Param/encoder.pt")
            torch.save(decoder.state_dict(),r"Param/decoder.pt")
            print("保存")





