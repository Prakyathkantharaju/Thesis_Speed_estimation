import pyxdf as xdf
import numpy as np
import torch
import pickle
import sys,os
import imufusion as im
from scipy import signal
import pandas as pd
def split(data,label,window=4,interval=0.1,sample_rate=148,vel_pos_flag=True,vert_acc_i=1):

    w_samples =window*sample_rate
    df=pd.DataFrame(data)
    for column in df.columns:
        if sum(df[column].isna())!=0:
            print(sum(df[column].isna())/len(df[column]))
            df[column]=df[column].fillna(method='backfill')

    X=df.to_numpy()
    X=np.array([data[-w_samples+i:i] for i in range(w_samples,data.shape[0]-w_samples,int(interval*sample_rate))])
    Y=np.ones(X.shape[0])*label
    
    if vel_pos_flag:
        b,a=signal.butter(5,20/(sample_rate/2),'low')
        vel=np.empty((X.shape[0],w_samples,1))
        pos=np.empty((X.shape[0],w_samples,1))
        velocity=np.empty((X.shape[0],w_samples,1))
        for i,sample in enumerate(X):
            vel[i,0,0]=0
            sample[:,vert_acc_i]=signal.filtfilt(b,a,sample[:,vert_acc_i])+981
            
            for j in range(X.shape[1]-1):
                vel[i,j+1,0]=vel[i,j,0]+sample[j,vert_acc_i]*(1/sample_rate)
            #plt.plot(vel[i,:,:])
            #plt.show()
            vel[i,:,0]=signal.detrend(vel[i,:,0])
            pos[i,0,0]=0
            for j in range(X.shape[1]-1):
                pos[i,j+1,0]=pos[i,j,0]+vel[i,j,0]*(1/sample_rate)
            pos[i,:,0]=signal.detrend(pos[i,:,0])
            position=pos[i,:,0]
            peaks=signal.find_peaks(position,distance=80)[0]
            bottoms=signal.find_peaks(-position,distance=80)[0]

            try:
                y_apex=position[peaks[-1]]
                y_bottom=position[bottoms[bottoms<peaks[-1]][-1]]-0.6
                del_y=y_apex-y_bottom
                del_x=np.sqrt(110**2-(110-del_y)**2)*0.01
                del_t=-(bottoms[bottoms<peaks[-1]][-1]-peaks[-1])/sample_rate
                x_dot=del_x/del_t
            except:
                x_dot=0

            velocity[i,:,0]=np.ones(w_samples)*x_dot
        X=np.concatenate((X,vel,pos,velocity),axis=2)

    print(f"Shape of X is {X.shape} \n Label is {Y.shape} ")
        
    return X[10:,:,:],Y[10:]    



def preprocess():

    map={1:1,2:2,3:0,4:1,5:2,6:0,7:1,8:2,9:0,10:1,11:2,12:0}
    map={10:1,11:1,12:0,13:2}
    X=np.empty((1,2*200,6))
    Y=np.empty((1))

    for i in map.keys():
        streams,header=xdf.load_xdf(f"../../../Recordings/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-0{i}_eeg.xdf")
        for stream in streams:
            if stream['info']['name'][0]=='polar accel':
                data=stream['time_series'] 
        #data=d[:,[j for j in range(45,51)]]
        #data[:,1]=data[:,1]*981

        x,y=split(data,map[i],window=2,sample_rate=200,vert_acc_i=0)

        X=np.concatenate((X,x),axis=0)
        Y=np.concatenate((Y,y),axis=0)


    


    
    return X,Y


if __name__ == "__main__":

    #data=np.ones((10000,6))
    #label=1.25
    import matplotlib.pyplot as plt
    #X,y=split(data,label)
    X,Y=preprocess()
    print(X.shape,Y.shape)
    plt.figure()
    plt.plot(X[1000,:,7])
    plt.show()