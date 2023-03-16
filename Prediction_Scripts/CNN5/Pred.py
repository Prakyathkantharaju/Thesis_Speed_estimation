import torch
import numpy as np
import torch.nn as nn
import scipy.signal as signal
import sys,os
import pylsl 

import matplotlib.pyplot as plt


sys.path.append('../../models/Data_based_models/CNN5/')
sys.path.append('../polar_utils/')

#local imports

from lsl_imu import DataInlet,SetupStreams
from model2 import CNN

class Canvas():

    def __init__(self):
        plt.figure()
        plt.xlabel('time')
        plt.ylabel('walking speed (m/s)')
        plt.yticks(np.arange(0.2,2.0,0.2))
        plt.title('Walking Speed Estimation')
        plt.pause(0.001)

    def plot(self,data):
        plt.cla()
        plt.plot(data)
        plt.ylim([0.2,2.0])
        plt.pause(0.001)
        
        

class Prediction():

    def __init__(self,time_step=0.3,data_length=200,sampling_rate=200,r_number=0.25):

        self.time_step=time_step
        self.sampling_rate=sampling_rate
        self.model=CNN(num_classes=1)
        self.stream_info=pylsl.stream_info('Speed Estimation CNN','Marker',1,0)
        self.outlet=pylsl.stream_outlet(self.stream_info)
        self.chart=Canvas()
        self.acquisition=SetupStreams()
        self.buffer=[]
        self.first_flag=True
        self.model.load_state_dict(torch.load('../../models/Data_based_models/CNN5/model_3_16.h5'))
        self.output(r_number)

    def kalman(self,state,measurement,process_var=0.01**2,measurement_var=0.015**2):

        estimate=[[],[]]
        state[0],state[1]=state[0]+0,state[1]+process_var
        estimate[0],estimate[1]=(state[1]*measurement+measurement_var*state[0])/(state[1]+measurement_var),(state[1]*measurement_var)/(state[1]+measurement_var)
        state=estimate

        return state


    def round_nearest(self,x,a=0.25):
        return np.round(np.round(x/a)*a,2)

    def predict(self,data):
        self.model.eval()
        with torch.no_grad():
            output=model.forward_run(data)
        
        return output.reshape((-1)).to_numpy()

    def output(self,r_number=0.25):
        while True:
            try:
                if self.first_flag:
                    data=self.acquisition.get_data(200)
                    output=self.predict(data)
                    self.buffer.append(self.round_nearest(output,r_number))
                    self.state=[output,0.1**2]
                    self.chart.plot(self.buffer[-15:])

                else:
                    data=self.acquisition.get_data(200)
                    output=self.predict(data)
                    sate=self.kalman(state,output[0])
                    self.buffer.append(self.round_nearest(state[0],r_number))
                    self.chart.plot(self.buffer[-15:])
            except KeyboardInterrupt:
                print("Ending Speed Estimation")
                break



if __name__=="__main__":
    main=Prediction()
