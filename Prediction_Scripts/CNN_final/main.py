import numpy as np
import torch 
import torch.nn as nn
import sys,os
import pickle
import pylsl
import matplotlib.pyplot as plt

sys.path.append('../../models/Data_based_models/CNN4/')
sys.path.append('../polar_utils/')


from model2 import CNN
from lsl_imu import DataInlet,SetupStreams



class Canvas():
    def __init__(self):
        plt.figure(figsize=(15,15))
        plt.ylabel('Speed (m/s)')
        plt.xlabel('Time (s)')
        plt.title('Speed Prediction')
        plt.pause(0.001)

    def plot_data(self,data):

        plt.cla()
        plt.plot(data,label='CNN 1 s')
        plt.yticks(np.arange(0.2,1.8,0.2))
        plt.legend()
        plt.pause(0.001)


def round_nearest(x,a=0.01):
    return np.round(np.round(x/a)*a,2)

def kalman(state,measurement,process_var=0.02**2,measurement_var=0.05**2):
        estimate=[[],[]]
        state[0],state[1]=state[0]+0,state[1]+process_var
        estimate[0],estimate[1]=(state[1]*measurement+measurement_var*state[0])/(state[1]+measurement_var),(state[1]*measurement_var)/(state[1]+measurement_var)
        state=estimate

        return state


def laod_model():
    model = CNN(input_features=3,input_length=200,num_classes=1)
    model.load_state_dict(torch.load('../../models\Data_based_models\CNN4\model_saves\model_3_19.h5'))
    model.eval()
    return model

def get_data(acquisition,normalizer,data_length=200):
    data=acquisition.get_data(data_length)
    data=normalizer.transform(data.reshape(1,-1))
    return data.reshape(1,3,data_length)


def predict(model,data):
    with torch.no_grad():
        pred=model.forward_run(data)
    return pred.item()

def main():
    model=laod_model()
    normalizer=pickle.load(open('normalizer.pickle','rb'))
    acquisition=SetupStreams()
    canvas=Canvas()
    speeds=[]
    state=[0,0]
    firt_time=True

    while True:
        if firt_time:
            data=acquisition.get_data(200)
            if data.shape[0]>=200:
                data=get_data(acquisition,normalizer)
                pred=predict(model,data)
                sate=[pred,0.1**2]
                firt_time=False

        else:
            data=get_data(acquisition,normalizer)
            pred=predict(model,data)
            state=kalman(state,pred)
            speeds.append(state[0])
            canvas.plot_data(speeds[-15:])
            print(state[0])        


if __name__ == "__main__":
    main()