import torch
import numpy as np
from positional_encodings.torch_encodings import PositionalEncodingPermute1D
import torch.nn as nn
import matplotlib.pyplot as plt

class Transformer(nn.Module):

    def __init__(self,in_features=3,in_length=200,encoder_layers=2,dim_forward=2048,num_classes=1):

        super(Transformer,self).__init__()

        self.conv1=nn.Conv1d(in_channels=3,out_channels=64,kernel_size=5,stride=1)
        self.batchnorm1=nn.BatchNorm1d(64)
        self.pos_enc=PositionalEncodingPermute1D(64)
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=196,nhead=14,dim_feedforward=dim_forward,batch_first=True)
        self.encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=encoder_layers)
        self.conv2=nn.Conv1d(64,128,kernel_size=7,stride=1)
        self.conv3=nn.Conv1d(128,256,kernel_size=7,stride=1)
        self.conv4=nn.Conv1d(256,128,kernel_size=7,stride=1)
        self.dense1=nn.Linear(128*2,num_classes)
        self.maxpool=nn.MaxPool1d(7,stride=3)
        self.batchnorm2=nn.BatchNorm1d(128)
        self.batchnorm3=nn.BatchNorm1d(256)
        self.batchnorm4=nn.BatchNorm1d(128)
        self.activation=nn.ReLU()
        self.device=torch.device('cpu')

    def forward(self,x):
        
        x=self.conv1(x)
        x=self.batchnorm1(x)

        x=self.pos_enc(x)
        x=self.activation(x)

        x=self.encoder(x)
        
        x=self.conv2(x)
        x=self.batchnorm2(x)
        x=self.activation(x)
        x=self.maxpool(x)

        x=self.conv3(x)
        x=self.batchnorm3(x)
        x=self.activation(x)
        x=self.maxpool(x)

        x=self.conv4(x)
        x=self.batchnorm4(x)
        x=self.activation(x)
        x=self.maxpool(x)
        
        x=self.dense1(x.reshape(x.shape[0],-1))
        
        return x.reshape(-1)

    def forward_run(self,x):
        
        x=torch.from_numpy(x.reshape(-1,self.in_features,self.in_length)).float()
        x=x.to(self.device)
        x=self.forward(x)

        return self.activation(x)
        



if __name__=="__main__":
    x=torch.randn(10,3,200)
    t=Transformer()
    x=t(x)



