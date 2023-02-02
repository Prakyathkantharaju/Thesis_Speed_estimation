import pickle, copy
from numpy.core.defchararray import array
import pandas as pd
import numpy as np
import sys, logging,os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,Normalizer
import itertools
# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal
# sklearn imports
from sklearn.model_selection import train_test_split
import time
#mlflow import
import pickle

# local imports
from dataset import IMU
from model import Transformer
from preprocessing import preprocess_data

# set the device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X,Y=preprocess_data()
X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
X=Normalizer().fit_transform(X)
X=X.reshape(X.shape[0],1,X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.55, random_state=42)

test_loss=[]
train_loss=[]
test_mse=[]
train_mse=[]
train_dataset=IMU(X_train,y_train)
test_dataset=IMU(X_test,y_test)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)

model=Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4)
criterion = nn.MSELoss()
n_total_steps = len(train_loader)
num_epochs=10
prev_time=time.time()
for epoch in range(num_epochs):
    model.train()
    train_labels=[]
    train_preds=[]
    test_labels=[]
    test_preds=[]
    print(f"Epoch {epoch+1} of {num_epochs}")
    for i, (inputs, labels) in enumerate(train_loader):
       
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss.append(loss.item())
        if time.time()-prev_time>100:
            print(f"Epoch {epoch+1} of {num_epochs} Step {i+1} of {n_total_steps}")
            prev_time=time.time()
            print(outputs)   
        # Backward and optimize
            print (f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if (i + 1) % 2 == 0: 
        train_labels+=labels.detach().cpu().tolist()
        train_preds+=outputs.detach().cpu().tolist()
    with torch.no_grad():
        model.eval()
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if time.time()-prev_time>100:
                print(f"Epoch {epoch+1} of {num_epochs} Step {i+1} of {n_total_steps}")
                prev_time=time.time()
                print(outputs)
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())
            test_labels+=labels.detach().cpu().tolist()
            test_preds+=outputs.detach().cpu().tolist()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {loss.item():.4f}')
           
    filename=f"model_{epoch+1}.h5"
    torch.save(model.state_dict(),filename)

    comparision_data={'train_labels':train_labels,'train_preds':train_preds,'test_labels':test_labels,'test_preds':test_preds}
    with open(f"comparision_data_{epoch+1}.pickle",'wb') as f:
        pickle.dump(comparision_data,f)