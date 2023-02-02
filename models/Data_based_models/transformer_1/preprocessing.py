import torch
import numpy as np
import pickle as pkl



def preprocess_data():
    with open("../../data/concat_data copy.pickle","rb") as f:
        data=pkl.load(f)

    x=data["X"]
    y=data["Y"]
    x=np.array(x)
    print(x[0].shape)

    X=[]
    Y=[]

    for i in range(len(x)-1):
        try:
            X.append(x[i].reshape(3,400))
            Y.append(y[i])
        except:
            #print(x[i].shape)
            pass
    X=np.array(X)
    Y=np.array(Y)

    return X[:,0,:].reshape(X.shape[0],1,X.shape[2]),Y.reshape(Y.shape[0],1)

if __name__=="__main__":
    X,Y=preprocess_data()
    print(X.shape)
    print(Y.shape)

    