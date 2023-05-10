import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    def __init__(self,in_features,out_features):
        self.dense1=nn.Linear(in_features=in_features,out_features=in_features//1.25)
        self.tanh=nn.Tanh()
        self.dense2=nn.Linear(in_features=in_features//1.25,out_features=out_features)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=x.reshape(x.shape[0],x.shape[2],x.shape[1])
        x=self.dense1(x)
        x=self.tanh(x)
        x=self.dense2(x)
        x=self.softmax(x)
        return x





class CNN(nn.Module):
    def __init__(self,input_features=1,input_lengt=200):
        super(CNN,self).__init__()
        self.input_features=input_features
        self.input_length=input_lengt
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1=nn.Conv1d(in_channels=input_features,out_channels=64,kernel_size=7,stride=1)
        self.attention1=Attention(in_features=64,out_features=32)
        self.conv2=nn.Conv1d(in_channels=64,out_channels=128,kernel_size=7,stride=1)
        self.attention2=Attention(in_features=128,out_features=64)
        self.conv3=nn.Conv1d(in_channels=128,out_channels=256,kernel_size=7,stride=1)
        self.attention3=Attention(in_features=256,out_features=128)
        self.dense1=nn.Linear(in_features=256*3,out_features=512)
        self.dense2=nn.Linear(in_features=512,out_features=1)

    def forward(self,x):
        x=self.conv1(x)
        x1=self.attention1(x)
        print(x1.shape)
        x1=x*x1
        print(x1.shape)
        x=self.conv2(x)
        x2=self.attention2(x)
        x2=x*x2
        x=self.conv3(x)
        x3=self.attention3(x)
        x3=x*x3
        x=torch.cat((x1,x2,x3),dim=1)
        x=self.dense1(x.reshape(x.shape[0],-1))
        x=self.dense2(x)

        return x

        




    