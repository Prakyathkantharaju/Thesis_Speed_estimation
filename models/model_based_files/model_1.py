import numpy as np
import matplotlib.pyplot as plt
from Speed_Recognition_3 import SpeedRec
import imufusion
import scipy as sc
from scipy.signal import find_peaks
import sys
# This calss needs to pull samples as soon as possible and update the buffer of IMU data
sys.path.append("../../")
class SpeedRec_1(SpeedRec):
    def __init__(self,h=110,c=0.4,data_length=800,sample_rate=200):
        super(SpeedRec_1, self).__init__(data_length,sample_rate,h,c)
        self.buffer_length=sample_rate*20*60
        self.data_length = data_length
        self.ahrs=imufusion.Ahrs()
        self.linear_buffer=np.empty((self.buffer_length,3))
        self.euler_buffer=np.empty((self.buffer_length,3))
        self.velocity_buffer=np.empty(400)
        self.position_buffer=np.empty(400)
        self.velocity_buffer[0]=0
        self.position_buffer[0]=0
        self.velocity_buffer[0],self.position_buffer[0]=0,0
        self.buffer_pointer=0
        self.vel_pointer=1
        self.pos_pointer=1
        self.vertical_acc_index=2
        self.ahrs.settings=imufusion.Settings(0.55,10,20,5*self.sampling_rate)
        self.split_counter=0
        self.vel_split_buffer=np.empty([self.buffer_length,400])
        self.pos_split_buffer=np.empty([self.buffer_length,400])
        self.split_pointer=0

    def ahrs_update(self,sample):
        #self.ahrs.update(sample)
        self.ahrs.update_no_magnetometer(sample[[0,1,2]],sample[[3,4,5]],1/self.sampling_rate)
        self.linear_buffer[self.buffer_pointer,:]=self.ahrs.linear_acceleration
        self.euler_buffer[self.buffer_pointer,:]=self.ahrs.quaternion.to_euler()
        self.buffer_pointer+=1
        if self.buffer_pointer>=self.buffer_length:
            self.buffer_pointer=0
            self.linear_buffer=np.empty((self.buffer_length,3))
            self.euler_buffer=np.empty((self.buffer_length,3))
        
        return None

    def  vel_pos_reset(self):
        self.vel_pointer=1
        self.pos_pointer=1
        self.velocity_buffer=np.empty(400)
        self.position_buffer=np.empty(400)
        self.velocity_buffer[0]=0
        self.position_buffer[0]=0
        self.split_counter=0
        return None

    def vel_update(self,buffer_pointer):
        if not self.ahrs.flags.initialising:
            self.velocity_buffer[self.vel_pointer]=self.velocity_buffer[self.vel_pointer-1]+self.linear_buffer[buffer_pointer,self.vertical_acc_index]*(1/self.sampling_rate)
            #self.position_buffer[self.vel_pos_pointer]=self.position_buffer[self.vel_pos_pointer-1]+self.velocity_buffer[self.vel_pos_pointer]*(1/self.sampling_rate)  
            self.vel_pointer+=1
        
        return None
    def pos_update(self,pointer):
        
        if not self.ahrs.flags.initialising:
            self.position_buffer[self.pos_pointer]=self.position_buffer[self.pos_pointer-1]+self.velocity_buffer[pointer]*(1/self.sampling_rate)
            self.pos_pointer+=1

        return None

    def update(self,sample):
        self.ahrs_update(sample)
        self.split_counter+=1
        if self.buffer_pointer>399 and self.split_counter>50:
            self.vel_pos_reset()
            for i in range(self.buffer_pointer-399,self.buffer_pointer):
                self.vel_update(i)
            self.velocity_buffer=sc.signal.detrend(self.velocity_buffer)
            for i in range(399):
                self.pos_update(i)
            self.position_buffer=sc.signal.detrend(self.position_buffer)
            self.vel_split_buffer[self.split_pointer] = self.velocity_buffer
            self.pos_split_buffer[self.split_pointer] = self.position_buffer
            self.split_pointer+=1
            

        return None
    
    def velocity1(self,rounding_numbers=0.2):
        if self.buffer_pointer<400:

            return None

        position=self.pos_split_buffer[self.split_pointer]
        self.peaks=find_peaks(position,distance=80)[0]
        self.bottoms=find_peaks(-1*position,distance=80)[0]
        
        try:
            y_apex=position[self.peaks[-1]]
            y_bottom = position[self.bottoms[self.bottoms < self.peaks[-1]][-1]]-self.c
            del_y=y_apex-y_bottom
            delta_x = np.sqrt(self.h**2-(self.h-del_y)**2)*0.01
            delta_t = -(self.bottoms[self.bottoms < self.peaks[-1]][-1] - self.peaks[-1]) /self.sampling_rate
            x_dot = self.round_nearest(delta_x / delta_t,rounding_numbers) 
            self.buffer.append(x_dot)
            self.delta_t=delta_t
            self.delta_x=delta_x
            self.delta_y=del_y
            return x_dot
        except:
            x_dot=0
            self.buffer.append(x_dot)
            return x_dot

    
    def Output(self,sample):
        self.update(sample)
        x_dot=self.velocity(rounding_numbers=0.2)
        self.buffer.append(x_dot)
        
   


        

   

        


    



