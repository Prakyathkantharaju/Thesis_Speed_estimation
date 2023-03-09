import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split,SubsetRandomSampler,ConcatDataset
from sklearn.model_selection import KFold
#import wandb 
#wandb.login()

def train_epoch(model,device,dataloader,loss_fn,optimzer):
    train_loss,train_correct=0.0,0
    model.train()
#    wandb.watch(model,loss_fn,log="all",log_freq=10)
    for images,labels in dataloader:

        images,labels=images.to(device),labels.to(device)
        optimzer.zero_grad()
        output=model(images)
        loss=loss_fn(output,labels)
        loss.backward()
        optimzer.step()
        train_loss+=loss.item()*images.size(0)
        scores,predictions=torch.max(output.data,1)
        train_correct+=(predictions==labels).sum().item()


    return train_loss,train_correct


def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss,valid_correct=0.0,0
    valid_labels,valid_outputs=[],[]
    model.eval()

    for images,labels in dataloader:

        images,labels=images.to(device),labels.to(device)
        output=model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores,predictions=torch.max(output.data,1)
        valid_correct+=(predictions==labels).sum().item()
        valid_labels+=labels.detach().cpu().tolist()
        valid_outputs+=predictions.detach().cpu().tolist()

    return valid_loss,valid_correct,valid_labels,valid_outputs



