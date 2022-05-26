import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['Kaiti']
from sklearn.metrics import r2_score


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)
         
        #输入通道数，输出通道数，卷积核大小
        #通道注意力，一维的卷积（1*1*1*D），D是通道数，1应该是卷积核个数，1是卷积核尺寸
        self.fc1   = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)  
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()  #sigmoid输出每个通道的权重
        
        #self.we1 = 0
    def forward(self, x):
        
        #两个并行的路线：特征基于宽度和高度的全局池化，
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        scale = self.sigmoid(out)  
        
        result = x*scale   
        #self.we1 = scale
        return result,scale
    
#空间注意力机制
#通道注意力的输出作为本模块的输入
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        #输入通道数，输出通道数，核尺寸
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        #self.we2 = 0
        self.relu = nn.ReLU()
    def forward(self, x):
        #channel attention的两个全局池化
        avg_out = torch.mean(x, dim=1, keepdim=True) #shape(inchannel,1)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x1 = torch.cat([avg_out, max_out], dim=1)  
        x1 = self.conv1(x1)    
        #print(x.shape)
        
        #print(x1.shape)
        x1 = x1.view(x1.size(0),9)
        scale = self.softmax(x1)
        #print(x1[0])
        #print(scale[5])
        scale = scale.view(scale.size(0),1,3,3)
        result = x*scale   
        #self.we2 =scale
        return  result,scale
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, ratio, no_spatial=False):
        super(CBAM, self).__init__()
        self.ratio = ratio
        self.ChannelAttention= ChannelAttention(gate_channels, ratio)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialAttention= SpatialAttention()
    def forward(self, x):
        x_out,we1 = self.ChannelAttention(x)
        if not self.no_spatial:
            x_out,we2 = self.SpatialAttention(x_out)
        return x_out,we1,we2


class STA_CNN_LSTM(nn.Module):
    def __init__(self):
        super(STA_CNN_LSTM,self).__init__()
        self.cbam = CBAM(64, 8)
        
        self.conv1 = nn.Conv2d(
            in_channels=3,     
            out_channels=32,  
            kernel_size=2,   
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=2,
            stride=1,
            padding=0         
        )

        self.convx = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=2,
            stride=1,
            padding=1         
        )
        
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        
        self.fc1 = nn.Linear(576,12)
        #self.fc2 = nn.Linear(64,64)
        self.dropout = nn.Dropout(p=0.5)
        
        self.lstm = nn.LSTM(input_size=3, hidden_size=64,
                            num_layers=3, batch_first=True)
        self.fc3 = nn.Sequential(nn.Linear(64,64),nn.ReLU())
        
        self.fc4 = nn.Sequential(nn.Linear(24, 4))#,nn.ReLU(),nn.Linear(128, 15))
        
        self.f1=nn.Linear(128,64)
        self.f2=nn.Linear(128,128)
        self.lstmcell=nn.LSTMCell(3,64)
        self.Softmax=nn.Softmax(dim=1)
        f1 = nn.Linear(128,12)
        
        self.fc0 = nn.Linear(128,1)
        
        self.afc1 = nn.Linear(128,12,bias = False)
        self.afc20 = nn.Linear(12,12)
        self.afc21 = nn.Linear(12,12)
        self.afc22 = nn.Linear(12,12)
        self.afc23 = nn.Linear(12,12)
        #self.afc24 = nn.Linear(12,12)
        #self.afc25 = nn.Linear(12,12)
        self.afc30 = nn.Sequential(nn.Sigmoid(),nn.Linear(12,1))
        self.afc31 = nn.Sequential(nn.Sigmoid(),nn.Linear(12,1))
        self.afc32 = nn.Sequential(nn.Sigmoid(),nn.Linear(12,1))
        self.afc33 = nn.Sequential(nn.Sigmoid(),nn.Linear(12,1))

        self.active = nn.Softmax(dim=1)
        
        self.Softmaxt=nn.Softmax(dim=1)
        self.lstmcell2=nn.LSTMCell(input_size=64, hidden_size=64)
        self.f11=nn.Sequential(nn.Linear(128,64),nn.ReLU())
        self.f22=nn.Linear(64,12)
        
    def Atten(self,h,c,x,xt):  #H(106,13,128)
        st = torch.cat((h,c),1)
        w1 = self.afc1(st)
        e = torch.zeros((x.size(0),3,1))
        for j in range(3):
            if j ==0:
            #print(x[:,:,j].shape)
                w2 = self.afc20(x[:,:,j].squeeze(1))
                w3 = self.afc30((w1+w2))
            if j ==1:
            #print(x[:,:,j].shape)
                w2 = self.afc21(x[:,:,j].squeeze(1))
                w3 = self.afc31((w1+w2))
            if j ==2:
            #print(x[:,:,j].shape)
                w2 = self.afc22(x[:,:,j].squeeze(1))
                w3 = self.afc32((w1+w2))*0.8


            #print(w3.shape)
            e[:,j] = w3
            #print(w3)
        atten = self.active(e)
        #print(atten.shape)
        #print(x[:,i+1,:].shape,a.shape)
        xt = xt*(atten.squeeze(2))

        return xt,atten
    
    def Atten_t(self,H):  
        h=H[:,-1,:].unsqueeze(1) 
        H=H[:,-1-12:-1,:] 
        #tem = torch.cat((H,h),1)
        atten=torch.matmul(h,H.transpose(1,2)).transpose(1,2)  
        #注意力矩阵,矩阵乘法，以及转置
        atten=self.Softmax(atten)
        atten_H=atten*H #带有注意力的历史隐状态  
        atten_H=torch.sum(atten_H,dim=1).unsqueeze(1) 
        return torch.cat((atten_H,h),2).squeeze(1),atten 

    def forward(self,x1,x2):
        x = F.relu(self.batchnorm1(self.conv1(x1)))
        X = F.relu(self.batchnorm2(self.conv2(x)))
        X = F.max_pool2d(X,kernel_size=(2,2),stride=(1,1),padding=1)  
        X= self.dropout(X)
        #print(X.shape)
        X,self.we1,self.we2 = self.cbam(X)
        #print(X.shape)
        X2 = F.sigmoid(self.convx(x1))
        #print(X2.shape)
        X = F.relu(X+X2)
        X = X.view(X.size(0),-1)   
        
        X = F.relu(self.fc1(X))    
        
        #x = self.fc2(x)            
        X= self.dropout(X)
        
        #LSTM模型
        h = Variable(torch.zeros(
             x2.size(0), 64))
        
        c = Variable(torch.zeros(
             x2.size(0), 64))
        self.we3 = torch.zeros((12,x2.size(0),3,1))
        H =Variable(torch.zeros(12,x2.size(0),64))
        for i in range(12):
            
            #print(e.shape)
            if i ==0:
                xt_1 = x2[:,0,:]
            else:
                xt_1 = xt
            h, c = self.lstmcell(xt_1.squeeze(1), (h, c))
            if i==11:
                #print('*******')
                break
            H[i] = h
            xt = x2[:,i+1,:]
                #预测  还是一个LSTM层，上层输出()
            xt,self.we3[i]=self.Atten(h,c,x2,xt) #获取结合了注意力的隐状态
        H = H.view((x2.size(0),12, 64))
        #print(H[0,:,0])
        h=h.squeeze(0)  
        c=c.squeeze(0)  
        H_pre=torch.empty((h.shape[0],12,64*2))  
        #print(h1.shape)
        for i in range(12): #解码
            h_t, c_t = self.lstmcell2(h, (h, c))    #预测  还是一个LSTM层，上层输出()
            #h_t(106*128)
            
            H=torch.cat((H,h_t.unsqueeze(1)),1)   
            h_atten,self.we4=self.Atten_t(H) #获取结合了注意力的隐状态
            H_pre[:,i,:]=h_atten  #记录解码器每一步的隐状态
            h, c = h_t, c_t  # 将当前的隐状态与细胞状态记录用于下一个时间步 
        
        out = self.f22(self.f11(H_pre[:,-1,:])).squeeze(1)
        xx = torch.cat((X,out),1)
        #print(X.shape,out.shape)
        result = self.fc4(xx)                   
        return result
