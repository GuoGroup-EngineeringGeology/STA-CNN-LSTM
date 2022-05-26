import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_data(sheet_name):
    print("Loading data...")
    
    #读取样本数据
    dataset = pd.read_excel(r'C:\Users\何小何\NLP\新浦滑坡预测模型\数据处理\新浦滑坡位移.xlsx',sheet_name=sheet_name)
    #print(type(dataset))
    dataset=np.array(dataset)[:,1:]
    features = dataset[:, :]
    labels = dataset[:, :]
    
    return dataset,features,labels

from sklearn.preprocessing import StandardScaler
ss_x=StandardScaler()
ss_y=StandardScaler()
def normalize(mx,ss_x):  #可以用ss_x反归一化
    mx=ss_x.fit_transform(mx)
    return mx,ss_x

def re_trafor(mx,ss_x):
    if type(mx)=='numpy.ndarray':
        mx=ss_x.inverse_transform(mx)
    else:
        mx=mx.data.numpy()
        mx=ss_x.inverse_transform(mx)
    return mx

def cnn_data(data):
    n_feat=data.shape[1]
    n_num=data.shape[0]
    data1=np.zeros((n_num-1,1,4,4))
    labels=np.zeros((n_num-1,15))
    for i in range(n_num-1):
        data1[i,:]=np.reshape(data[i,:],(4,4))
        labels[i]=data[i+1,:-1]
    return data1,labels

def rnn_data(data):
    n_feat=data.shape[1]
    n_num=data.shape[0]
    step=12
    data1=np.zeros((n_num-step,step,n_feat))
    labels=np.zeros((n_num-step,n_feat-1))
    #for i in range(n_feat):
    for j in range(0,n_num-step):
        data1[j,:,:]=data[j:j+step,:]
        labels[j,:]=data[j+step,:-1]
        
    return data1,labels

def crnn_data(data):
    n_feat=data.shape[1]
    n_num=data.shape[0]
    step=12
    data1=np.zeros((n_num-step,12,4,4))
    labels=np.zeros((n_num-step,15))
    for i in range(n_num-step):
        data1[i,:]=np.reshape(data[i:i+step,:],(12,4,4))
        labels[i]=data[i+step,:-1]
    return data1,labels


def fig(y1,y2,num,label1,label2,title):
    t=range(num)
    #t=range(80)
    plt.figure(facecolor='white',figsize=(10,5))
    plt.plot(t,y1, 'r-', linewidth=2, label=label1)
    plt.plot(t, y2, 'g*-', linewidth=2, label=label2)
    
    plt.legend(loc='lower center')
    plt.legend(loc='lower center')
    plt.title(title, fontsize=18)
    plt.grid(b=True, ls=':')
    plt.show()

def mape(x,y,model,ss_y,num):
    x1=x.clone()
    y_per=model(x1)
    y_pre=y_per.data.numpy()
    y1=y.clone()
    y_train2=y1.data.numpy()
    
    y_pre1=ss_y.inverse_transform(y_pre)               #ss_x 三列  ss_X对x的两列  ss_y对y的一列
    y_tra=ss_y.inverse_transform(y_train2)

    y_pre2=np.reshape(y_pre1,num)
    y_tra1=np.reshape(y_tra,num)
    
    mape01 = np.sum((((y_pre2-y_tra1)/y_tra1)**2)**0.5)/num
    
    return y_pre2,mape01
