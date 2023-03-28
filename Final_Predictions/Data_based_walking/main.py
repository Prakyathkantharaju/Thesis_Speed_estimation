import torch
import numpy as np
import matplotlib.pyplot as plt
import pylsl
import sys,os
import pickle

sys.path.append('models/Data_based_models/CNN4/')
sys.path.append('Prediction_Scripts/polar_utils/')

from model2 import CNN
from lsl_imu import DataInlet,SetupStreams

class Prediction():
    def __init__(self,walking_index=2):
        self.buffer=[]
        self.filtered_buffer=[]
        self.walking_index=walking_index
        self.model=self.load_model()
        self.normalizer=self.load_normalizer()
        self.acquisition=SetupStreams()
        return None
    


    def load_model(self):
        model = CNN(input_features=3,input_length=200,num_classes=1)
        model.load_state_dict(torch.load('../../models/Data_based_models/CNN4/model_saves/model_3_19.h5'))
        model.eval()
        return model
    
    def load_normalizer(self):
        normalizer=pickle.load(open('../../models/Data_based_models/CNN4/normalizer.pickle','rb'))
        return normalizer
    
    def get_data(self,acquisition,normalizer,data_length=200):
        data=acquisition.get_data(data_length)
        if data.shape[0]!=data_length:
            return None
        data=normalizer.transform(data.reshape(1,-1))
        return data.reshape(1,3,data_length)
    
    def predict(self,model,data):
        with torch.no_grad():
            pred=model.forward_run(data)
        return pred.item()
    
    def round_nearest(self,x,a=0.01):
        return np.round(np.round(x/a)*a,2)
    
    def kalman(self,state,measurement,process_var=0.02**2,measurement_var=0.05**2):
        estimate=[[],[]]
        state[0],state[1]=state[0]+0,state[1]+process_var
        estimate[0],estimate[1]=(state[1]*measurement+measurement_var*state[0])/(state[1]+measurement_var),(state[1]*measurement_var)/(state[1]+measurement_var)
        state=estimate

        return state
    
    def reset_buffer(self):
        self.filtered_buffer=[]
        return None

    def output(self,model,normalizer,acquisition,Activity_pred):
        data=self.get_data(acquisition,normalizer)
        if data is None:
            return 0
        pred=self.predict(model,data)
        self.buffer.append(self.round_nearest(pred))
        if Activity_pred==self.walking_index:
            if len(self.filtered_buffer)==0:
                self.filtered_buffer.append(pred)
                self.state=[pred,0.1**2]
                return self.round_nearest(pred)
            else:
                self.state=self.kalman(self.state,pred)
                self.filtered_buffer.append(self.round_nearest(self.state[0]))
                return self.round_nearest(self.state[0])
        else:
            self.reset_buffer()
            return 0
              
        
        



