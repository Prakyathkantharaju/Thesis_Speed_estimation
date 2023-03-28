import torch
import numpy as np
import matplotlib.pyplot as plt
import pylsl
import sys,os
import pickle
import time

sys.path.append('models/Activity_recognition/CNN1/')
sys.path.append('Prediction_Scripts/polar_utils/')

from model import NeuralNet
from lsl_imu import DataInlet,SetupStreams


class prediction():
    def __init__(self, slider):
        input_size = 3 
        hidden_size = 100
        num_classes = 5
        dropout_rate= 0.3
        self.prev_flag = False
        self.device = torch.device('cpu')
        self.Activity_condition = {'RUNNING':0,
                              #'RUNNING_GRAD':1,
                              'WALKING':1,
                              'WALKING_GRAD':2,
                              'STAND':3,
                              'SQUATTING':4 }
        self.slider = slider
        self.stand = []
        self.mean_stand = np.array([0,0,0])
        model = NeuralNet(input_size, hidden_size, num_classes,dropout_rate).to(self.device)

        MODEL_PATH = 'models/Activity_recognition/CNN1/model_1_57_.h5'

        model.load_state_dict(torch.load(MODEL_PATH,  map_location=torch.device('cpu')))
        model.to(self.device)
        model.eval()

        self.model = model

        self.acquisition = SetupStreams()
        self.first_flag = True
        # acquisition
        # sys.exit()

    def mean_data(self,stand_flag):
            
        if stand_flag:
            data = self.acquisition.get_data(100)
            self.stand.append(data)
            #print("stand",np.array(self.stand).shape)
            self.prev_flag = True
        elif stand_flag is False and self.prev_flag:
            self.mean_stand = ((np.array(self.stand).reshape(-1,3)).mean(axis=0))
            #print("mean stand",self.mean_stand.shape)
            self.stand = []
            self.prev_flag = False


    def predict(self):
        prob_value = self.slider.value() / 100
        data = self.acquisition.get_data(200)            
        data = data - self.mean_stand 
        #print("data",data.shape, data.dtype,self.mean_stand)
        output = self.model.forward_run(data)
        _, max_predicted = torch.max(output.data, 1)
        probability_prediction = []
        for key,values  in self.Activity_condition.items():
            if output[0][values] > prob_value:
                probability_prediction.append(1)
            else:
                probability_prediction.append(0)
        probability_prediction = np.array(probability_prediction)
        return max_predicted.numpy(), probability_prediction, data





