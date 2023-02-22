import torch
import torch.nn as nn
from torch.autograd import Variable
class CNN_LSTM(nn.Module):
    """

    input will be (batch,input axis(3)(channels),data point)

    """
    def __init__(self,input_size=3,hidden_size=256,num_layers=4,num_classes=6,dropout_1=0.3,dropout_2=0.3,input_length=200):
        super(CNN_LSTM,self).__init__()
        self.input_size=input_size
        self.input_length=input_length
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.num_classes=num_classes
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device=torch.device('cpu')
        self.cnn1=nn.Conv1d(in_channels=input_size,out_channels=64,kernel_size=9)
        #lout=lin-(kernel-1)=192
        self.batch1=nn.BatchNorm1d(num_features=64)
        self.pool1=nn.MaxPool1d(3,stride=3)
        self.cnn2=nn.Conv1d(in_channels=64,out_channels=128,kernel_size=9)
        #lout=192-8=184
        self.batch2=nn.BatchNorm1d(num_features=128)
        self.dropout1=nn.Dropout(p=dropout_1)
        #input for lstm=(batch,128,184)
        #reshape (batch,184,128)
        self.LSTM=nn.LSTM(input_size=128,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout_2)
        #output=(batch,184,256(hidden_size))
        #reshape (batch,256,184)
        self.cnn3=nn.Conv1d(in_channels=hidden_size,out_channels=128,kernel_size=9)
        #lout=184-8=176
        self.batch3=nn.BatchNorm1d(num_features=128)
        self.dropout2=nn.Dropout(p=dropout_2)
        #self.dense1=nn.Linear(in_features=18*128,out_features=8*128)
        self.dense1=nn.Linear(in_features=128*3,out_features=num_classes)
        self.activation=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)

    
    def forward(self,x):
        h_0=Variable(torch.zeros(self.num_layers,x.shape[0],self.hidden_size)).to(self.device)
        c_0=Variable(torch.zeros(self.num_layers,x.shape[0],self.hidden_size)).to(self.device)
        
        x=self.cnn1(x)
        x=self.activation(x)
        x=self.pool1(x)
        x=self.batch1(x)
        #x=self.dropout1(x)
        x=self.cnn2(x)
        x=self.activation(x)
        x=self.pool1(x)
        x=self.batch2(x)
        #x=self.dropout2(x)
        #print(x.shape)
        x,_=self.LSTM(x.reshape(x.shape[0],x.shape[2],x.shape[1]),(h_0,c_0))
        #print(x.shape)
        x=self.cnn3(x.reshape(x.shape[0],x.shape[2],x.shape[1]))
        x=self.activation(x)
        x=self.pool1(x)
        x=self.batch3(x)
        #print(x.shape)
        y=self.dense1(x.reshape(x.shape[0],-1))
        # x=self.dropout2(x)
        # x=self.activation(x)
        # x=self.dense2(x)
        # y=self.dense3(x)

        return y.reshape(-1)

    def forward_run(self,x):
        x=torch.from_numpy(x.reshape(-1,self.input_size,self.input_length)).float()
        x=x.to(self.device)
        y=self.forward(x)
        return self.softmax(y)

if __name__ == '__main__':
    x=torch.randn(10,3,200)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=x.to(device)
    model=CNN_LSTM(3,256,5,4,0.3,0.5,200)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()))
    model(x)