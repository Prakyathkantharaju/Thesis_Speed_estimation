import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)

        div_term=1/(10000**((2*np.arange(d_model))/d_model))
        pe[:,0::2]=torch.sin(position*div_term[0::2])
        pe[:,1::2]=torch.cos(position*div_term[1::2])

        pe=pe.unsqueeze(0).transpose(0,1)

        self.register_buffer('pe',pe)


    def forward(self,x):
        return x + self.pe[:x.size(0),:].repeat(1,x.shape[1],1)





class Transformer(nn.Module):

    def __init__(self,num_features=3,feature_size=250,num_layers=3,num_classes=5):
        super(Transformer,self).__init__()
        self.input_embedding=nn.Linear(num_features,feature_size)
        self.src_mask=None

        self.pos_encoder=PositionalEncoding(feature_size)
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=feature_size,nhead=10,dropout=0.1)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=num_layers)
        self.decoder=nn.Linear(feature_size,32)
        self.decoder1=nn.Linear(200*32,num_classes)
        

    def init_weights(self):
        initrange=0.1
        self.decoder.bias.zero_()
        self.decoder.weight_data.uniform_(-initrange,initrange)

    def forward(self,src):
        print(src.shape)
        src=src.reshape(src.shape[1],src.shape[0],src.shape[2])
        src=self.input_embedding(src)
        src=self.pos_encoder(src)
        output=self.transformer_encoder(src)
        output=self.decoder(output)

        return self.decoder1(output.reshape((output.shape[1],output.shape[0]*output.shape[2])))

if __name__=="__main__":
    src=torch.rand((200,64,3))
    model=Transformer()
    out=model(src)
    print(out.shape)
