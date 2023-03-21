import torch
import torch.nn as nn
import numpy as np


"""
CNN structure 

input shape (batch,channels (features),length)

Out shape = ((Lin-(kernel -1) - 1)+2*padding/stride+1)
"""


class CNN(nn.Module):
    def __init__(self,input_features=3,input_length=200,num_classes=5,dropout=0.1):
        super(CNN,self).__init__()
        # variables
        self.in_channels=input_features
        self.in_length=input_length
        self.num_classes=num_classes
        self.dropout=nn.Dropout(dropout)
        self.activation=nn.ReLU()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        # Architecture

        self.conv1=nn.Conv1d(in_channels=self.in_channels,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.lout1=self.in_length-6
        # lout1=194
        self.batchnorm1= nn.BatchNorm1d(num_features=64)
        
        self.conv2=nn.Conv1d(in_channels=64,out_channels=128,kernel_size=7,stride=1,padding=2)
        self.lout2=self.lout1-6
        #lout2=188
        self.batchnorm2=nn.BatchNorm1d(num_features=128)

        self.conv3=nn.Conv1d(in_channels=128,out_channels=256,kernel_size=7,stride=1,padding=2)
        self.lout3=self.lout2-6
        #lout3= 182
        self.batchnorm3=nn.BatchNorm1d(num_features=256)

        self.conv4=nn.Conv1d(in_channels=256,out_channels=512,kernel_size=5,stride=1,padding=2)
        # lout = 174
        self.lout4=self.lout3-8
        self.maxpool1=nn.MaxPool1d(kernel_size=2,stride=2)
        # lout= 87
        self.lout4=(self.lout4-2)/2 + 1
        self.batchnorm4=nn.BatchNorm1d(num_features=512)

        self.conv5=nn.Conv1d(in_channels=512,out_channels=128,kernel_size=5,stride=1,padding=2)
        # lout = 78 
        self.lout5= self.lout4-9

        
        # lout = 39
        self.lout6=(self.lout5-2)/2 + 1
        self.batchnorm5=nn.BatchNorm1d(num_features=128)


        self.dense1=nn.Linear(in_features=int(128*2),out_features=num_classes)


        self.softmax=nn.Softmax(dim=1)


        
    def forward(self,x):

        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.activation(x)
        x=self.maxpool1(x)

        x=self.conv2(x)
        x=self.batchnorm2(x)
        x=self.activation(x)
        x=self.maxpool1(x)

        x=self.conv3(x)
        x=self.batchnorm3(x)
        x=self.activation(x)
        x=self.maxpool1(x)

        x=self.conv4(x)
        x=self.batchnorm4(x)
        x=self.activation(x)
        x=self.maxpool1(x)

        x=self.conv5(x)
        x=self.batchnorm5(x)
        x=self.activation(x)
        x=self.maxpool1(x)

        #print(x.shape)
        x=x.reshape((-1,x.shape[1]*x.shape[2]))
        x=self.dense1(x)

        

        return x.reshape((-1))


    def forward_run(self,x):
        x=torch.from_numpy(x.reshape(-1,self.in_channels,self.in_length)).float()
        x=x.to(self.device)
        y=self.forward(x)
        
        return self.activation(y)


if __name__=="__main__":
    model=CNN(num_classes=1)
    model.load_state_dict(torch.load("model_saves/model_1_15.h5"))
    torch.onnx.export(model,torch.randn(1,3,200),"model.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names=['input'],output_names=['output'],dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
    


        
