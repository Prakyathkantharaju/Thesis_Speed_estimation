import numpy as np
import imufusion as im
from Speed_Recognition_3 import SpeedRec as SP
from Speed_Recognition_3 import SpeedRecProcessing as SPP
import math
import scipy
from scipy import signal


class Speed_model_2(SP):
    def __init__(self,h=110,c=0.4,data_length=800,sample_rate=200):
        super(Speed_model_2,self).__init__(h=h,c=c,data_length=data_length,sample_rate=sample_rate)
        self.data_length=data_length
        self.sample_rate=sample_rate
        self.buffer_length=self.sample_rate*60*20
        self.ahrs=im.Ahrs()
        self.linear_buffer=np.empty((self.buffer_length,3))
        self.euler=np.empty((self.buffer_length,3))
        self.buffer_pointer=0

    def update(self,linear,gyro):
        self.ahrs.update_no_magnetometer(linear,gyro,1/self.sample_rate)
        self.linear_buffer[self.buffer_pointer,:]=self.ahrs.linear_acceleration
        self.euler[self.buffer_pointer,:]=self.ahrs.quaternion.to_euler()
        self.buffer_pointer+=1
        
        if self.buffer_pointer>self.buffer_length:
            self.linear_buffer=np.empty((self.buffer_length,3))
            self.euler=np.empty((self.buffer_length,3))
            self.buffer_pointer=0

        return None

    def Sim_update(self,linear_data,gyro_data):
        self.sim_length=len(linear_data)
        if self.sim_length>self.buffer_length:
            self.sim_length=self.buffer_length
            print("Total length of the data is more than buffer length, data length changed to buffer length")

        for i in range(self.sim_length):
            self.update(linear_data[i,:],gyro_data[i,:])

        return None

    def simulate_split_data(self,data,time_interval=4,split_interval=0.5,num_features=3):

        length=int(((len(data)/self.sample_rate)-time_interval)/split_interval)
        imu_buffer=np.empty((length,time_interval*self.sample_rate,num_features))
        for i in range(length):
            imu_buffer[i,:,:]=data[int(self.sample_rate*split_interval*i):int((split_interval*i+time_interval)*self.sample_rate),:]
        return imu_buffer

    def read_xdf(self,path):

        return None 

    def Simulation(self,raw_data=[],path="",vert_acc_index=2,data_flag=False):
        if not data_flag:
            IMU_data=self.read_xdf(path)
        else:
            IMU_data=np.array(raw_data)
        self.Sim_update(IMU_data[:,:3],IMU_data[:,3:])

        split_data=self.simulate_split_data(self.linear_buffer,num_features=3)        

        first_run=True
        data=split_data[:,:,2]*100
        first_data=data[0,:]

        if first_run:
            x_dot=self.Velocity(first_data)
            self.state=[x_dot,0.25**2]
            self.filtered_buffer.append(x_dot)
            first_run=False
        try:
            for d in data:
                x_dot=self.Velocity(d)
                self.state=self.Kalman_1D(self.state,x_dot)
                self.filtered_buffer.append(self.round_nearest(x_dot,a=0.05))
                self.TrendFilt(double_filt_flag=False)
                self.TrendFilt(double_filt_flag=True)

            values={"unfiltered":self.buffer,"kalman":self.filtered_buffer,"trend":self.Trend_buffer,"double":self.speeds}

            return values

        except:

            values={"unfiltered":self.buffer,"kalman":self.filtered_buffer,"trend":self.Trend_buffer,"double":self.speeds}
            return values
    
        

        
