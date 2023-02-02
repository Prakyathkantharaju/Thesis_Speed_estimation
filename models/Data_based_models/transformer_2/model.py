import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D


class Transformer(nn.Module):
    def __init__(self,input_dims=2,encoder_layers=4,dropout=0.1):
        super(Transformer,self).__init__()
        self.dropout=dropout
        self.input_dims=input_dims
        self.num_heads=16
        self.conv1=nn.Conv1d(input_dims,64,5,1,2)
        self.conv2=nn.Conv1d(64,128,5,1,2)
        self.conv3=nn.Conv1d(128,256,5,1,2)
        self.conv4=nn.Conv1d(256,128,5,1,2)
        self.conv5=nn.Conv1d(128,64,5,1,2)
        self.fc_1=nn.Linear(in_features=64*400,out_features=16*400)
        self.fc_2=nn.Linear(in_features=16*400,out_features=1)
        self.transformer_layer=nn.TransformerEncoderLayer(d_model=256,nhead=self.num_heads,dim_feedforward=2064,batch_first=True)
        self.transformer_encoder=nn.TransformerEncoder(self.transformer_layer,num_layers=encoder_layers)
        self.fc1=nn.Linear(in_features=51200*2,out_features=2064)
        self.fc2=nn.Linear(in_features=2064,out_features=256)
        self.fc3=nn.Linear(in_features=256,out_features=1)
        self.layer_norm=nn.LayerNorm([2064])
        self.relu=nn.ReLU()
        self.gelu=nn.GELU()

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        x=self.fc_1(x)
        x=self.relu(x)
        x=self.fc_2(x)

        #x=x.reshape(x.shape[0],x.shape[2],x.shape[1])
        #x=self.transformer_encoder(x)
        #x=x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        #print(x.shape)
        #x=x.reshape(x.shape[0],51200)
        #x=self.fc1(x)
        #x=self.layer_norm(x)
        #x=self.relu(x)
        #x=self.fc2(x)
        #x=self.relu(x)
        #x=self.fc3(x)
        return x
