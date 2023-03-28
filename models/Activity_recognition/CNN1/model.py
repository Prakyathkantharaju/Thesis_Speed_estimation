import torch
import torch.nn as nn



class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,dropout_rate_1: float,input_length=200):
        super(NeuralNet, self).__init__()
        self.device=torch.device('cpu')
        self.input_size = input_size
        self.input_length=input_length
        self.num_classes=num_classes
        # Extract features, 1D conv layers
        self.layer_1   = nn.Conv1d(input_size, 64, 5, stride=1,padding=2)
        self.activation_relu = nn.ReLU()
        self.layer_2 = nn.Conv1d(64, 64, 5, stride=1,padding=2)
        self.layer_3 = nn.Conv1d(64, 128, 5, stride=1,padding=2)
        self.layer_4 = nn.MaxPool1d(2, stride=2)
        self.layer_5 = nn.Dropout(p=0.2)
        self.layer_6 = nn.Conv1d(128, 128, 5, stride=1,padding=2)
        self.layer_7 = nn.Conv1d(128, 256, 5, stride=1,padding=2)
        self.layer_8 = nn.Conv1d(256, 256, 5, stride=1,padding=2)
#         self.layer_9 = nn.AvgPool1d(97)
        self.layer_10 = nn.Dropout(p=dropout_rate_1)
        self.layer_11 = nn.Linear(256, self.num_classes)
        self.activation_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation_relu(x)
        x = self.layer_4(x)
        #print(x.shape)
        x = self.layer_2(x)
        x = self.activation_relu(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        #print(x.shape)
        x = self.layer_3(x)
        x = self.activation_relu(x)
        #print(x.shape)
        x = self.layer_4(x)
        #print(x.shape)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.activation_relu(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        # print(x.shape)
        # x = self.layer_6(x)
        # x = self.activation_relu(x)
        # x = self.layer_4(x)
        x = self.layer_7(x)
        x = self.activation_relu(x)
        x = self.layer_4(x)
        #print(x.shape)
        x = self.layer_8(x)
        x = self.activation_relu(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        #print(x.shape)
        self.layer_9 = nn.AvgPool1d(x.shape[2])
        x = self.layer_9(x)
        #print(x.shape)
        x = self.layer_10(x)
        y = self.layer_11(x.reshape(x.shape[0],-1))
        #print(y.shape)
        return y


    def forward_run(self, x):
        # print(x, x.shape, type(x))
        x = torch.from_numpy(x.reshape(-1, self.input_size, self.input_length)).float()
        x=x.to(self.device)
        # print(x, x.shape, type(x))
        y = self.forward(x)
        return self.activation_softmax(y)

if __name__ == '__main__':
    x=torch.randn(10,3,200)
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device('cuda')
    x=x.to(device)
    model=NeuralNet(3,100,6,0.3)
    print(sum(p.numel() for p in model.parameters()))
    model = model.to(device)
    model(x)
    #print(model(x))