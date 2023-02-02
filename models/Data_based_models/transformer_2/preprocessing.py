import torch
import numpy as np
import pickle as pkl
from scipy import signal

import matplotlib.pyplot as plt
def preprocess_data():
    with open("../../data/concat_data.pickle","rb") as f:
        data=pkl.load(f)

    x=data["X"]
    y=data["Y"]
    h=data["h"]
    x=np.array(x)
    #print(x[0].shape)
    #plt.plot(x[100][:,0])
    #plt.show()
    X=[]
    Y=[]
    H=[]
    for i in range(len(x)-1):
        if x[i].shape[0]!=400:
            continue
            
            #X.append(x[i].reshape(3,400))

            
        else:
            X.append(x[i])
            #print(x[i].shape)
            Y.append(y[i])
            H.append(h[i])
    H=np.array(H)
    #print(H.shape[0][0])  
    X=np.array(X)
    #print(X.shape)
    Y=np.array(Y)
    Y=Y*100
    x_=[]
    for i in range(len(X)):
        a,b,c,d=[],[],[],[]
        for j in range(400):
            a.append(X[i][j][0])
            b.append(X[i][j][1])
            c.append(X[i][j][2])
            d.append(H[i][j])
        x_.append(np.array([a,b,c,d]))
    X=np.array(x_)
    #print(X.shape)
    print(X[0].shape)
    #plt.plot(X[500,0,:])
    #plt.show()
    return X[:,[0,3],:].reshape(X.shape[0],2,X.shape[2]),Y.reshape(Y.shape[0],1)



def Feature_extraction():
    X,Y=preprocess_data()
    x=[]
    fs=200
    fc=20
    w=fc/(fs/2)
    b,a=signal.butter(5,w,'low')

    for sample in X:
        sub_x=[]
        velocity=[0]
        position=[0]
        sample=sample.reshape(400)
        #plt.plot(sample)
        #plt.show()
        sample=signal.filtfilt(b,a,sample)
        for i in range(len(sample)-1):
            v=velocity[i]+sample[i]*(1/200)
            velocity.append(v)
        velocity=signal.detrend(velocity)
        for i in range(len(velocity)-1):
            p=position[i]+velocity[i]*(1/200)
            position.append(p)
        position=signal.detrend(position)
        #plt.plot(sample)
        #plt.show()
        sub_x.append(sample)
        sub_x.append(velocity)
        sub_x.append(position)
        x.append(sub_x)
        #print(np.array(sub_x[2]).shape)

    x=np.array(x)

    #print(x.shape)
    return x,Y

if __name__=="__main__":
    import matplotlib.pyplot as plt
    #X,Y=Feature_extraction()
    X,Y=preprocess_data()
    #plt.plot(X[0])
    #plt.show()


