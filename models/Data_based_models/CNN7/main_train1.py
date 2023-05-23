import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb
#sklearn imports

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset,DataLoader,random_split,SubsetRandomSampler


# local imports

from dataset import IMU
from model2 import CNN
#from model import NeuralNet
from preprocessing1 import preprocess
from data_prep import train_epoch,valid_epoch



# Data prep
X,Y=preprocess()
X=X.reshape(-1,X.shape[2]*X.shape[1])
norm=Normalizer()
norm.fit(X)
pickle.dump(norm,open('normalizer.pickle','wb'))
X=norm.transform(X)
X=X.reshape(-1,1,100)
dataset=IMU(X,Y,size=100,num_features=1)

# model load

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

criterion=nn.MSELoss()

num_epochs=50
batch_size=64
k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

history={'train_loss':[],'test_loss':[],'train_acc':[],'test_acc':[],'labels':[],'outputs':[]}


for fold,(train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print(f'Fold : {fold}')

        train_sampler=SubsetRandomSampler(train_idx)
        test_sampler=SubsetRandomSampler(val_idx)
        train_loader=DataLoader(dataset,batch_size=batch_size,sampler=train_sampler)
        test_loader=DataLoader(dataset,batch_size=batch_size,sampler=test_sampler)
        
        model=CNN(input_features=1,num_classes=1)
        model.to(device)
        optimizer=optim.Adam(model.parameters(),lr=3.5e-4)

        for epoch in range(num_epochs):
            #wandb.watch(model, log="all", log_freq=10)
            train_loss,train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss,test_correct,test_labels,test_outputs=valid_epoch(model,device,test_loader,criterion)

            train_loss=train_loss/len(train_loader.sampler)
            train_acc=train_correct/len(train_loader.sampler)*100
            test_loss=test_loss/len(test_loader.sampler)
            test_acc=test_correct/len(test_loader.sampler)*100

            print(f"epoch {epoch+1}/{num_epochs} , avg training loss : {train_loss} , training acc : {train_acc} , avg testing loss : {test_loss} ,testing acc : {test_acc} ]")

            #wandb.log({'fold':fold,"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['labels'].append(test_labels)
            history['outputs'].append(test_outputs)


            pickle.dump(history,open('history.pickle','wb'))

            if epoch>15 and test_loss<0.003:
                torch.save(model.state_dict(),f'model_{fold}_{epoch}.h5') 
