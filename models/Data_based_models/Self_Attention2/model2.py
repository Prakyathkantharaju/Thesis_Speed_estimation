import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    def __init__(self,in_features,out_features):
        super(Attention,self).__init__()
        self.dense1=nn.Linear(in_features=in_features,out_features=int(in_features//1.25))
        self.dense2=nn.Linear(in_features=int(in_features//1.25),out_features=out_features)
        self.softmax=nn.Softmax(dim=1)
        self.tanh=nn.Tanh()

    def forward(self,x):
        x=x.reshape(x.shape[0],x.shape[2],x.shape[1])
        x=self.dense1(x)
        x=self.tanh(x)
        x=self.dense2(x)
        x=self.softmax(x)
        return x





class CNN(nn.Module):
    def __init__(self,input_features=1,input_lengt=200,num_classes=1):
        super(CNN,self).__init__()
        self.input_features=input_features
        self.input_length=input_lengt
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1=nn.Conv1d(in_channels=self.input_features,out_channels=64,kernel_size=7,stride=2)
        self.batchnorm1=nn.BatchNorm1d(num_features=64)
        self.relu1=nn.ReLU()
        self.dropout1=nn.Dropout(p=0.1)
        self.conv2=nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5,stride=2)
        self.batchnorm2=nn.BatchNorm1d(num_features=128)
        self.attention1=Attention(in_features=128,out_features=16)
        self.conv3=nn.Conv1d(in_channels=128,out_channels=256,kernel_size=5,stride=2)
        self.batchnorm3=nn.BatchNorm1d(num_features=256)
        self.attention2=Attention(in_features=256,out_features=16)
        self.conv4=nn.Conv1d(in_channels=256,out_channels=128,kernel_size=5,stride=2)
        self.batchnorm4=nn.BatchNorm1d(num_features=128)
        self.attention3=Attention(in_features=128,out_features=16)
        self.dense1=nn.Linear(in_features=8192,out_features=512)
        self.dense2=nn.Linear(in_features=512,out_features=num_classes)

        # To save the extracted features
        self.f_conv1=None
        self.f_conv2=None
        self.f_conv3=None
        self.f_conv4=None

        self.f_attention1=None
        self.f_attention2=None
        self.f_attention3=None

    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.relu1(x)
        self.f_conv1=x

        x=self.conv2(x)
        x=self.batchnorm2(x)
        x=self.relu1(x)
        x=self.dropout1(x)
        self.f_conv2=x

        x1=self.attention1(x)
        x1=torch.bmm(x,x1)
        self.f_attention1=x1

        x=self.conv3(x)
        x=self.batchnorm3(x)
        x=self.relu1(x)
        self.f_conv3=x

        x2=self.attention2(x)
        x2=torch.bmm(x,x2)
        self.f_attention2=x2

        x=self.conv4(x)
        x=self.batchnorm4(x)
        x=self.relu1(x)
        self.f_conv4=x

        x3=self.attention3(x)
        x3=torch.bmm(x,x3)
        self.f_attention3=x3
        #x=torch.flatten(x,start_dim=1)
        x1,x2,x3=torch.flatten(x1,start_dim=1),torch.flatten(x2,start_dim=1),torch.flatten(x3,start_dim=1)
        x=torch.cat((x1,x2,x3),dim=1)

        x=self.dense1(x)
        x=self.relu1(x)
        x=self.dropout1(x)
        x=self.dense2(x)

        return x.reshape(-1)

        

if __name__=='__main__':
    model=CNN()
    x=torch.randn(32,1,100)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.MSELoss()
    optimizer.zero_grad()
    output=model(x)
    loss=criterion(output,torch.randn(32,1))
    loss.backward()
    optimizer.step()
    print(loss.item())    


    