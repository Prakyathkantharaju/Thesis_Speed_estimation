import os,sys
import numpy as np
import torch 
import pickle   
import pylsl
import time
from lsl_imu import DataInlet,SetupStreams
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler,Normalizer
sys.path.append("../models/Data_based_models/LSTM1/")
sys.path.append("../moldels/Data_based_models/LSTM1/")
sys.path.append("../moldels/Data_based_models/LSTM2/")

from model2 import CNN_LSTM as LSTM2
from model import CNN_LSTM as LSTM1
#from transformer_1.model2 import CNN_LSTM as LSTM3
from Speed_Recognition_3 import SpeedRec 


class Canvas():
    def __init__(self):
        plt.figure(figsize=(15,15))
        plt.ylabel('Speed (m/s)')
        plt.xlabel('Time (s)')
        plt.title('Speed Prediction')
        plt.pause(0.001)

    def plot_data(self,old,model_1,model_2):
        plt.cla()
        plt.plot(old,label='Actual')
        plt.plot(model_1,label='LSTM1')
        plt.plot(model_2,label='LSTM2')
        plt.yticks(np.arange(0.2,1.8,0.2))
        plt.legend()
        plt.pause(0.001)

def round_nearest(x,a):
    if math.isnan((x / a) * a):
        return 0
    else:
        num=np.round(np.round(x / a) * a, -int(math.floor(math.log10(a))))
        if num==1.2:
            num=1.25
        return num

def split(data,label,window=4,interval=0.3,sample_rate=148,vel_pos_flag=False,vert_acc_i=1):
    w_samples =window*sample_rate
    X=np.array([data[-w_samples+i:i] for i in range(w_samples,data.shape[0]-w_samples,int(interval*sample_rate))])

    return X

def model_2_convert(pred):
    if pred==0:
        pred=1
    elif pred==1:
        pred=1.25
    elif pred==2:
        pred=1.5
    elif pred==3:
        pred=0.5
    elif pred==4:
        pred=0.75
     
    return pred
def prediction():
    acquisition=SetupStreams()
    first_run=True
    prev_time=time.time()   
    Normalizer=pickle.load(open("../models/Data_based_models/LSTM3/normalizer.pkl","rb"))
    model1=LSTM1(input_size=3,num_classes=1,input_length=200)
    model2=LSTM2(input_size=3,num_classes=5,input_length=200)
    model1.load_state_dict(torch.load("New_models/LSTM4_10.h5"))
    model2.load_state_dict(torch.load("New_models/LSTM3_10.h5"))
    model1.eval()
    model2.eval()
    SRP=SpeedRec(h=110,c=1.5,data_length=400)
    chart=Canvas()
    model1_speeds=[]
    model2_speeds=[]
    while True:
        if first_run:
            data=acquisition.get_data(400)
            
            if data.shape[0]>399:
                first_run=False
                print(data.shape,'hre')
                speed=SRP.Output(data[:,0],compare_mode=True,rounding_number=0.25)
                X=data.reshape(2,3,200)
                X=X.reshape(2,600)
                norm_data=Normalizer.fit_transform(X)
                #print(norm_data.shape)
                norm_data=norm_data.reshape(2,3,200)
                norm_data=norm_data[1,:,:].reshape(1,3,200)
                prev_time=time.time()
                with torch.no_grad():
                    pred1=model1.forward_run(norm_data)
                    pred2=model2.forward_run(norm_data)
                    _,pred2=torch.max(pred2,1)
                    pred1=round_nearest(pred1.item(),0.25)
                    model1_speeds+=[pred1]
                    num=pred2.detach().cpu()
                    model2_speeds+=[model_2_convert(num)]
                    chart.plot_data(SRP.speeds[-15:],model1_speeds[-15:],model2_speeds[-15:])
                    print(f"old:{SRP.speeds[-1]} new:{model1_speeds[-1]} new2:{model2_speeds[-1]}")
        else:
            data=acquisition.get_data(400)
            if (time.time()-prev_time)>0.5:
                speed=SRP.Output(data[:,0],compare_mode=True,rounding_number=0.25)
                X=data.reshape(2,3,200).reshape(2,600)
                norm_data=Normalizer.fit_transform(X).reshape(2,3,200)
                norm_data=norm_data[1,:,:].reshape(1,3,200)
                #print(norm_data.shape)
                norm_data=norm_data.reshape(1,3,200)
                prev_time=time.time()
                with torch.no_grad():
                    pred1=model1.forward_run(norm_data)
                    pred2=model2.forward_run(norm_data)
                    _,pred2=torch.max(pred2,1)
                    pred1=round_nearest(pred1.item(),0.25)
                    model1_speeds+=[pred1]
                    num=pred2.detach().cpu()
                    model2_speeds+=[model_2_convert(num)]
                    chart.plot_data(SRP.speeds[-15:],model1_speeds[-15:],model2_speeds[-15:])
                    print(f"old:{SRP.speeds[-1]} new:{model1_speeds[-1]} new2:{model2_speeds[-1]}")



if __name__=="__main__":
    prediction()
        


