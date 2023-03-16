import torch.nn as nn
import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute1D
#encoder_layer=nn.TransformerEncoderLayer(d_model=3,nhead=3,dim_feedforward=1024,batch_first=True)
#encoder=nn.TransformerEncoder(encoder_layer,num_layers=3)
#pos_enc=PositionalEncoding1D(3)
#src=torch.randn(10,20,3)
#src_encoded=pos_enc(src)
#src=src.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#out=encoder(src_encoded)
#print(out.shape)


class Transformer(nn.Module):
    def __init__(self,input_dims=9,conv_out=64,num_heads=8,dim_ff=1024,encoder_layers=3,dropout=0.1):
        super(Transformer,self).__init__()
        self.dropout=dropout
        self.conv_out=conv_out
        self.input_dims=input_dims
        self.num_heads=num_heads
        self.pos_encoder=PositionalEncodingPermute1D(conv_out)
        self.conv1=nn.Conv1d(input_dims,conv_out,5,stride=1,padding=2)
        self.lout1=296
        #lin=200, lout=200
        self.TransformerEncoderLayer=nn.TransformerEncoderLayer(d_model=conv_out,nhead=num_heads,dim_feedforward=dim_ff,batch_first=True)
        self.TransformerEncoder=nn.TransformerEncoder(self.TransformerEncoderLayer,num_layers=encoder_layers)
        self.fc1=nn.Linear(in_features=conv_out*self.lout1,out_features=2064)
        self.fc2=nn.Linear(in_features=2064,out_features=256)
        self.fc3=nn.Linear(in_features=256,out_features=3)
        self.batchnorm1=nn.BatchNorm1d(2064)
        self.batchnorm2=nn.BatchNorm1d(256)
        self.relu=nn.ReLU()
        self.layernorm1=nn.LayerNorm([2064])
        
    def forward(self,x):
        x=x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        x=self.conv1(x)
        #print(x.shape)
        x=self.pos_encoder(x)
        x=x.reshape(x.shape[0],x.shape[2],x.shape[1])
        x=self.TransformerEncoder(x)
        #print(x.shape)
        x=x.reshape(x.shape[0],self.lout1*self.conv_out)
        x=self.fc1(x)
        #x=self.layernorm1(x)
        x=self.batchnorm1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.batchnorm2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x


if __name__=="__main__":
    model=Transformer()
    model=model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    x=torch.randn(10,8,296)
    x=x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(model(x))


        
        
