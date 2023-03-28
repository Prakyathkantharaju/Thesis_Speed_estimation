import torch
import numpy as np
import matplotlib.pyplot as plt
import pylsl
import sys,os
import pickle

from Final_Predictions.Data_based_walking.main import Prediction as Walking_Prediction
from Final_Predictions.Activity_Recognition.act_main import prediction as Activity_Prediction

from UI.ui1 import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

class App():
    def __init__(self):
        self.app=QApplication(sys.argv)
        self.ui=Ui_MainWindow()
        self.MainWindow=QMainWindow()
        self.ui.setupUi(self.MainWindow)
        self.standing=self.ui.label
        self.squatting=self.ui.label_3
        self.walking=self.ui.label_5
        self.running=self.ui.label_7
        self.label_list=[self.standing,self.squatting,self.walking,self.running]
        for label in self.label_list:
            label.setStyleSheet("background-color: red")

        self.act_pred=Activity_Prediction(self.ui.horizontalSlider)
        self.walking_pred=Walking_Prediction()
        self.act_buffer=[]
        self.walk_buffer=[]
        self.timer=QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.main_pred)
        self.timer.start()

    def act_out(self):

        self.act_pred.mean_data(self.ui.pushButton.isChecked())
        pred_max,pred_prob,data=self.act_pred.predict()
        index=np.where(pred_prob>0)[0]

        if index[0]==2:
            index[0]=1
        elif index[0]==3:
            index[0]=2
        elif index[0]==4:
            index[0]=3
            pass

        for i in range(4):
            if i==index[0]:
                self.label_list[i].setStyleSheet("background-color: green")
            else:
                self.label_list[i].setStyleSheet("background-color: red")

        if index.size==0:
            self.act_buffer.append(np.NAN)
        else:
            self.act_buffer.append(index[0])

        return index[0]
    
    def main_pred(self):
        act_index=self.act_out()
        speed=self.walking_pred.predict(act_index)
        self.walk_buffer.append(speed)

        self.ui.figure1.clear()
        ax=self.ui.figure1.add_subplot(111)
        ax.plot(self.walk_buffer[-15:])
        ax.set_ylim(0,2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Walking Speed (m/s)')
        self.ui.canvas1.draw()




        
