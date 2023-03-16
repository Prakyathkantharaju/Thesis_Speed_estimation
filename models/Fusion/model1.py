import numpy as np
import scipy.signal as singal
import os,sys
import pylsl
# local imports 
sys.path.append("../Data_based_models/LSTM6/")
from model2 import CNN_LSTM




class Model_Update():

    def __init__(self,h=110,data_length=800,c=0.5):
        self.h=h
        self.data_length=data_length
        self.sample_rate=200
        self.b,self.a=singal.butter(5,'low')
        self.time=np.arange(0,self.data_length/self.sample_rate,1/self.sample_rate)
        
        self.position=[]
        self.delta_x=[]
        self.delta_t=[]
        self.delta_y=[]
        self.velocity=[]
        self.parameter=[]
        self.avg_speed=[]
        self.buffer=[]
        self.avg_speed_trend=[]
        self.avg_speed=[]    
        self.filtered_buffer=[]
        self.ukf_speed=[]
        self.Trend_buffer=[]
        self.x_cov=[]
        self.state=[]
        self.speeds=[]
        #self.kf=KalmanFilter(dim_x=1,dim_z=1)
        self.fs=self.sampling_rate
        self.fc=20
        self.w=self.fc/(self.fs/2)
        self.b,self.a=signal.butter(5,self.w,'low')
        self.h=h
        self.c=c
        self.state=[]
 
    def threshold(self,x,a):
        if math.isnan((x / a) * a):
            return 0
        else:
            return np.round((x // a) * a, -int(math.floor(math.log10(a))))

    def round_nearest(self,x,a):
        if math.isnan((x / a) * a):
            return 0
        else:
            return np.round(np.round(x / a) * a, -int(math.floor(math.log10(a))))

    def Velocity_tuning(self,data,rounding_numnber=0.02):
        rounding_numnbers=0.02
        velocity=[0]
        position=[0]
        self.data=data+450
        self.acc=signal.filtfilt(self.b,self.a,self.data)
        #self.acc=self.data
        for i in range(0,len(self.acc)-2):
            v=velocity[i]+self.acc[i]*(self.time[i+1]-self.time[i])
            velocity.append(v)
        velocity=signal.detrend(velocity)
        for i in range(0,len(velocity)-2):
            p=position[i]+velocity[i]*(self.time[i+1]-self.time[i])
            position.append(p)
        position=signal.detrend(position)
        self.position=position
        self.velocity=velocity
        self.peaks=find_peaks(position,distance=80)[0]
        self.bottoms=find_peaks(-1*position,distance=80)[0]
        try:
            y_apex=position[self.peaks[-1]]
            y_bottom = position[self.bottoms[self.bottoms < self.peaks[-1]][-1]]-self.c
            del_y=y_apex-y_bottom
            delta_x = np.sqrt(self.h**2-(self.h-del_y)**2)*0.01
            delta_t = -(self.bottoms[self.bottoms < self.peaks[-1]][-1] - self.peaks[-1]) /200 
            x_dot = self.round_nearest(delta_x / delta_t,rounding_numnbers) 
            self.buffer.append(x_dot)
            self.delta_t=delta_t
            self.delta_x=delta_x
            self.delta_y=del_y
            return x_dot
        except:
            x_dot=0
            self.buffer.append(x_dot)
            return x_dot

    def iterative_c(self,e,init_c=0.25,init_alpha=0.5,decay_rate=0.0001,iteration=0):
        alpha=init_alpha/(1+decay_rate*iteration)
        new_c=init_c+alpha*e
        self.c=new_c

        return new_c


    def Kalman_1D(self,state,measurement,process_var=0.005**2,measurement_var=0.1**2):
        estimate=[[],[]]
        state[0],state[1]=state[0]+0,state[1]+process_var
        estimate[0],estimate[1]=(state[1]*measurement+measurement_var*state[0])/(state[1]+measurement_var),(state[1]*measurement_var)/(state[1]+measurement_var)
        state=estimate

        return state

 


    

