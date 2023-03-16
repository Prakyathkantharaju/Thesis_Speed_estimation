import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb
#sklearn imports

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import KFold,train_test_split
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
X=X.reshape(-1,3,200)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
train_dataset=IMU(X_train,Y_train,size=200,num_features=3)
test_dataset=IMU(X_test,Y_test,size=200,num_features=3)
print(f'train samples : {len(train_dataset)}, test samples : {len(test_dataset)}')
# model load

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

criterion=nn.MSELoss()

num_epochs=20
batch_size=64
k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)

history={'train_loss':[],'test_loss':[],'train_acc':[],'test_acc':[],'test_labels':[],'test_outputs':[],'val_loss':[],'val_acc':[],'val_labels':[],'val_outputs':[]}


for fold,(train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):

        print(f'Fold : {fold}')

        train_sampler=SubsetRandomSampler(train_idx)
        val_sampler=SubsetRandomSampler(val_idx)
        train_loader=DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler)
        val_loader=DataLoader(train_dataset,batch_size=batch_size,sampler=val_sampler)
        test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
        
        model=CNN(num_classes=1)
        model.to(device)
        optimizer=optim.Adam(model.parameters(),lr=3.5e-4)

        for epoch in range(num_epochs):
            #wandb.watch(model, log="all", log_freq=10)
            train_loss,train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            val_loss,val_correct,val_labels,val_outputs=valid_epoch(model,device,val_loader,criterion)
            test_loss,test_correct,test_labels,test_outputs=valid_epoch(model, device, test_loader, criterion)

            train_loss=train_loss/len(train_loader.sampler)
            train_acc=train_correct/len(train_loader.sampler)*100
            val_loss=val_loss/len(val_loader.sampler)
            val_acc=val_correct/len(val_loader.sampler)*100
            test_loss=test_loss/len(test_loader.sampler)
            test_acc=test_correct/len(test_loader.sampler)*100


            print(f"epoch {epoch+1}/{num_epochs} , avg training loss : {train_loss} , training acc : {train_acc} , avg val loss : {val_loss},val acc : {val_acc},avg testing loss : {test_loss} ,testing acc : {test_acc} ]")

            #wandb.log({'fold':fold,"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_labels'].append(test_labels)
            history['test_outputs'].append(test_outputs)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_labels'].append(val_labels)
            history['val_outputs'].append(val_outputs)


            pickle.dump(history,open('history.pickle','wb'))

            if epoch>1 and test_acc>=history['test_acc'][-2]:
                torch.save(model.state_dict(),f'model_{fold}_{epoch}.h5') 
