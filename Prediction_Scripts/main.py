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
#sys.path.append("../models/Data_based_models/LSTM1/")
#sys.path.append("../moldels/Data_based_models/LSTM1/")
#sys.path.append("../moldels/Data_based_models/LSTM2/")

sys.path.append('CNN_models/')
import pylsl
from model2 import CNN as CNN2
from model import CNN as CNN1
#from model3 import CNN_LSTM as LSTM3
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
        plt.plot(old,label='Old Model')
        plt.plot(model_1,label='CNN 1 s')
        plt.plot(model_2,label='CNN 2 s')
        plt.yticks(np.arange(0.2,1.8,0.2))
        plt.legend()
        plt.pause(0.001)

def round_nearest(x,a=0.01):
    return np.round(np.round(x/a)*a,2)

def split(data,label,window=4,interval=0.3,sample_rate=148,vel_pos_flag=False,vert_acc_i=1):
    w_samples =window*sample_rate
    X=np.array([data[-w_samples+i:i] for i in range(w_samples,data.shape[0]-w_samples,int(interval*sample_rate))])

    return X

def Kalman_1D(state,measurement,process_var=0.01**2,measurement_var=0.05**2):
        estimate=[[],[]]
        state[0],state[1]=state[0]+0,state[1]+process_var
        estimate[0],estimate[1]=(state[1]*measurement+measurement_var*state[0])/(state[1]+measurement_var),(state[1]*measurement_var)/(state[1]+measurement_var)
        state=estimate

        return state

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
    else:
        pred=0
    return pred

def speed_2_label(pred):
    if pred==1:
        pred=0
    elif pred==1.25:
        pred=1
    elif pred==1.5:
        pred=2
    elif pred==0.5:
        pred=3
    elif pred==0.75:
        pred=4

     
    return torch.tensor(pred)


def streams():
    info_old=pylsl.stream_info('old','Marker',1,0,'float32')
    info_new=pylsl.stream_info('0.5 s model','Marker',1,0,'float32')
    info_new2=pylsl.stream_info('1 s model','Marker',1,0,'float32')
    outlet_old=pylsl.stream_outlet(info_old)
    outlet_new=pylsl.stream_outlet(info_new)
    outlet_new2=pylsl.stream_outlet(info_new2)

    return outlet_old,outlet_new,outlet_new2

def prediction():
    acquisition=SetupStreams()
    first_run=True
    prev_time=time.time()   
    Normalizer=pickle.load(open("../models/Data_based_models/LSTM6/normalizer.pickle","rb"))
    model1=CNN1(num_classes=1,input_length=200)
    model2=CNN2(num_classes=1,input_length=100)
    #model3=LSTM3(input_size=3,num_classes=5,input_length=100)
    model1.load_state_dict(torch.load("models/Data_based_models/CNN4/model_saves/model_0_18.h5"))
    model2.load_state_dict(torch.load("models/Data_based_models/CNN7/model_saves/model_3_29.h5"))
    
    model1.eval()
    model2.eval()
    SRP=SpeedRec(h=110,c=1.5,data_length=400)
    chart=Canvas()
    model1_speeds=[]
    model2_speeds=[]
    flag=False
    outlet_old,outlet_new,outlet_new2=streams()
    while True:
        if first_run:
            data=acquisition.get_data(400)
            
            if data.shape[0]>399:
                flag=True
                first_run=False
                print(data.shape,'hre')
                speed=SRP.Output(data[:,0],compare_mode=True,rounding_number=0.25)
                X=data.reshape(2,3,200)
                X=X.reshape(2,600)
                norm_data=Normalizer.fit_transform(X)
                #print(norm_data.shape)
                norm_data=norm_data.reshape(2,3,200)
                norm_data=norm_data[1,:,:].reshape(1,3,200)
                norm_data1=norm_data.reshape(2,3,100)[1,:,:].reshape(1,3,100)
                prev_time=time.time()
                with torch.no_grad():
                    pred1=model1.forward_run(norm_data)
                    pred2=model2.forward_run(norm_data1)
                    #_,pred2=torch.max(pred2,1)
                    #_,pred1=torch.max(pred1,1)
                    num1=pred1.detach().cpu()
                    #pred1=round_nearest(pred1.item(),0.25)
                    model1_speeds+=[round_nearest(num1)]
                    num=pred2.detach().cpu()
                    model2_speeds+=[round_nearest(num)]
                    state1=[num1,0.1**2]
                    state2=[num,0.1**2]
                    chart.plot_data(SRP.speeds[-15:],model1_speeds[-15:],model2_speeds[-15:])
                    print(f"old:{SRP.speeds[-1]} new:{model1_speeds[-1]} new2:{model2_speeds[-1]}")
        else:
            data=acquisition.get_data(400)
            if (time.time()-prev_time)>0.5:
                flag=True
                speed=SRP.Output(data[:,0],compare_mode=True,rounding_number=0.25)
                X=data.reshape(2,3,200).reshape(2,600)
                norm_data=Normalizer.fit_transform(X).reshape(2,3,200)
                norm_data=norm_data[1,:,:].reshape(1,3,200)
                #print(norm_data.shape)
                norm_data1=norm_data.reshape(2,3,100)[1,:,:].reshape(1,3,100)
                norm_data=norm_data.reshape(1,3,200)
                prev_time=time.time()
                with torch.no_grad():
                    out1=model1.forward_run(norm_data)
                    out2=model2.forward_run(norm_data1)
                    #_,pred1=torch.max(out1,1)
                    #_,pred2=torch.max(out2,1)
                    #print(out2[0,pred2])
                    #pred1=round_nearest(pred1.item(),0.25)
                    num1=pred1.detach().cpu()
                    state1=Kalman_1D(state1,num1)
                    model1_speeds+=[round_nearest(state1[0])]
                    num=pred2.detach().cpu()
                    state2=Kalman_1D(state2,num)
                    model2_speeds+=[round_nearest(state2[0])]

#                    model1_speeds+=[model_2_convert(num1)]
#                    num=pred2.detach().cpu()
#                    model2_speeds+=[model_2_convert(num)]
                    chart.plot_data(SRP.speeds[-15:],model1_speeds[-15:],model2_speeds[-15:])
                    print(f"old:{SRP.speeds[-1]} new:{model1_speeds[-1]} new2:{model2_speeds[-1]}")
        if len(SRP.speeds)>0 and flag:
            outlet_old.push_sample([SRP.speeds[-1]])
            outlet_new.push_sample([model1_speeds[-1]])
            #print(model1_speeds[-1])
            outlet_new2.push_sample([model2_speeds[-1]])
            flag=False


if __name__=="__main__":
    prediction()
        


