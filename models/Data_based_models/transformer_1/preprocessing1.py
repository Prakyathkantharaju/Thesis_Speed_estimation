import pyxdf as xdf
import numpy as np
import torch
import pickle
import sys,os
import imufusion as im
from scipy import signal
def split(data,label,window=2,interval=0.3,sample_rate=148,vel_pos_flag=True,vert_acc_i=1):

    w_samples =window*sample_rate
    X=np.array([data[-w_samples+i:i] for i in range(w_samples,data.shape[0]-w_samples,int(interval*sample_rate))])
    Y=np.ones(X.shape[0])*label

    if vel_pos_flag:
        b,a=signal.butter(5,20/(sample_rate/2),'low')
        vel=np.empty((X.shape[0],w_samples,1))
        pos=np.empty((X.shape[0],w_samples,1))
        for i,sample in enumerate(X):
            vel[i,0,0]=0
            sample[:,vert_acc_i]=signal.filtfilt(b,a,sample[:,vert_acc_i])
            for j in range(X.shape[1]-1):
                vel[i,j+1,0]=vel[i,j,0]+sample[j,vert_acc_i]*(1/sample_rate)
            vel[i,:,:]=signal.detrend(vel[i,:,:])
            pos[i,0,0]=0
            for j in range(X.shape[1]-1):
                pos[i,j+1,0]=pos[i,j,0]+vel[i,j,0]*(1/sample_rate)
            pos[i,:,0]=signal.detrend(pos[i,:,0])
                

        X=np.concatenate((X,vel,pos),axis=2)

    print(f"Shape of X is {X.shape} \n Label is {Y.shape} ")
        
    return X,Y    



def preprocess():



    IMU_data=np.empty()
    
    return None


if __name__ == "__main__":

    data=np.ones((10000,6))
    label=1.25

    X,y=split(data,label)
    print(X.shape)
