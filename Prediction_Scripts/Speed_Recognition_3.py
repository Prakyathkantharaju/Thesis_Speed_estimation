import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import pylsl
import pyxdf as xdf

# class Canvas():
#     def __init__(self):
       
#         plt.figure()
#         plt.ylabel('speed (m/s)')
#         plt.figure(figsize=(15,15))
#         plt.pause(0.001)

#     def plot_data(self,data):
#         plt.cla()
#         plt.plot(data)
#         plt.yticks(ticks = [0.8, 1, 1.25, 1.5, 1.75], label = [0.8, 1.0, 1.25, 1.5,1.75] , fontsize=20)
#         plt.ylim([0.2,2])
#         plt.pause(0.001)
    
#     def compare_plot(self,unfiltered,kalman,trend,double):
#         #data=[unfiltered,kalman,trend,double]
#         legends=['Unfiltered','Kalman','Trend','Double Filtered']
#         plt.cla()
#         plt.plot(unfiltered,alpha=0.3)
#         plt.plot(kalman,alpha=0.5)
#         plt.plot(trend,alpha=0.6)
#         plt.plot(double)
#         plt.yticks(ticks = [0.8, 1, 1.25, 1.5, 1.75], label = [0.8, 1.0, 1.25, 1.5,1.75] , fontsize=20) 
#         plt.ylabel('Speed (m/s)') 
#         plt.ylim([0.75,1.8])  
#         plt.legend(legends)
#         plt.pause(0.001)

#     def parameter_plot(self,data):
#         plt.cla()
#         plt.plot(data)
#         plt.pause(0.001)
        # for id,d in enumerate(data):
        #     plt.subplot(4,1,(id+1))
        #     plt.cla()
        #     plt.plot(d)
        #     plt.ylabel('Speed (m/s)')
        #     plt.yticks(ticks = [0.8, 1, 1.25, 1.5, 1.75], label = [0.8, 1.0, 1.25, 1.5,1.75] , fontsize=10)
        #     plt.legend(legends[id])
        #     plt.pause(0.001)
            

class SpeedRec():
    def __init__(self,data_length=800,sample_rate=200,h=110,c=0.4):
        self.data_length=data_length
        self.time=np.arange(0,self.data_length/sample_rate,1/sample_rate)
        self.sampling_rate=sample_rate
        info=pylsl.stream_info("Speed_stream","Marker",1,0,'float32','myuidw43537')
        self.outlet=pylsl.stream_outlet(info)
        #self.chart=Canvas()
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
        self.Qt=0.005
        self.Rt=0.005

    # def threshold(self,x,a):
    #     if math.isnan((x / a) * a):
    #         return 0
    #     else:
    #         return np.round((x // a) * a, -int(math.floor(math.log10(a))))



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

    def Lateral_velocity(self,lateral_acc,peak_idx,bottom_idx):
        velocity=[0]
        lat_vel=0        
        self.lat_acc=signal.filtfilt(self.b,self.a,lateral_acc)
        for i in range(0,len(self.lat_acc)-2):
            v=velocity[i]+self.lat_acc[i]*(self.time[i+1]-self.time[i])
            velocity.append(v)
        for v in velocity[peak_idx, bottom_idx]:
            lat_vel+=v
        lat_vel=lat_vel/(peak_idx-bottom_idx)

        return lat_vel



    def Velocity(self,data,rounding_numnber=0.02):
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


    def Kalman_1D(self,state,measurement,process_var=0.005**2,measurement_var=0.1**2):
        estimate=[[],[]]
        state[0],state[1]=state[0]+0,state[1]+process_var
        estimate[0],estimate[1]=(state[1]*measurement+measurement_var*state[0])/(state[1]+measurement_var),(state[1]*measurement_var)/(state[1]+measurement_var)
        state=estimate

        return state

    def UKF(self,cc,Qt,Rt,x_dot_prev,h,n):

        lambd=2
        xm=cc
        X=np.zeros([1,2*n+1])
        # sigma points
        X[0,0]=xm
        X[0,1]=xm+(n*self.x_cov)**2
        X[0,2]=xm-(n*self.x_cov)**2

        # weights
        w=np.zeros([1,2*n+1])
        w[0,0]=lambd/(n+lambd)
        w[0,1]=(n)/(2*(n+lambd))
        w[0,2]=(n)/(2*(n+lambd))

        # computing new mean and covariance of x
        xm=np.sum(X,axis=1)/(2*n+1)
        #print(xm)
        self.x_cov=0
        for i in range(0,2*n+1):
            self.x_cov=self.x_cov+w[0,i]*(X[0,i]-xm)**2+Rt
        
        # calculating measurement from model
        Z=np.zeros([1,2*n+1])
        for i in range(0,2*n+1):
            self.delta_y[-1]=self.delta_y[-1]-self.c+X[0,i]
            Z[0,i]=((0.01*(((2*h*self.delta_y)-(self.delta_y*self.delta_y))**0.5)))/self.delta_t

        # calculating new mean and covariance of Z
        Zm=np.sum(Z,axis=1)/(2*n+1)
        Z_cov=0
        for i in range(0,2*n+1):
            Z_cov=Z_cov+w[0,i]*(Z[0,i]-Zm)**2+Qt

        # calculating Kalman gain
        xz_cov=0
        for i in range(0,2*n+1):
            xz_cov=xz_cov+w[0,i]*(Z[0,i]-Zm)*(X[0,i]-xm)
        Kg=xz_cov/Z_cov

        # calculating new parameter
        xm=xm+Kg*(x_dot_prev-Zm)
        z=Zm
        self.c=xm
        self.ukf_speed.append(self.round_nearest(z,0.05))
        self.parameter.append(self.c)
        return cc

    def TrendFilt(self,sh_buf_ln=5,filter_num=2,double_filt_flag=True,r_number=0.2):
        if double_filt_flag:
            
            if len(self.filtered_buffer)<20:
                self.avg_speed=self.filtered_buffer[-1]
                self.speeds.append(self.avg_speed)
            else:
                s_buffer=self.filtered_buffer[-sh_buf_ln:]
                if len([i for i in s_buffer if i >self.avg_speed])>filter_num or len([i for i in s_buffer if i <self.avg_speed])>filter_num:
                    self.avg_speed=self.threshold(np.nanmean(self.filtered_buffer[-3:]),r_number)
                    self.speeds.append(self.avg_speed)
                else:
                    self.speeds.append(self.avg_speed)
        else:
            
            if len(self.buffer)<20:
                self.avg_speed_trend=self.buffer[-1]
                self.Trend_buffer.append(self.avg_speed_trend)
            else:
                sh_buffer=self.buffer[-sh_buf_ln:]
                if len([i for i in sh_buffer if i >self.avg_speed_trend])>filter_num or len([i for i in sh_buffer if i <self.avg_speed_trend])>filter_num:
                    self.avg_speed_trend=self.threshold(np.nanmean(self.buffer[-3:]),r_number)
                    self.Trend_buffer.append(self.avg_speed_trend)
                else:
                    self.Trend_buffer.append(self.avg_speed_trend)
        return self.avg_speed_trend
            

    def Output(self,data,first_run=False,rounding_number=0.2,Kalman_flag=False,trend_flag=True,double_filt_flag=False,compare_mode=False,UKF_flag=False):
        x_dot=self.Velocity(data,0.02)
        #print(x_dot)
        # Kalman_flag=main_plotting.kal_click()
        # trend_flag=main_plotting.trend_click()
        # UKF_flag=main_plotting.ukf_click()
        
        if(Kalman_flag and trend_flag):
            double_filt_flag=True
            Kalman_flag=False
            trend_flag=False
        
        if compare_mode:

            if len(self.buffer)==1:
                self.state=[x_dot,0.15**2]
                self.filtered_buffer.append(x_dot)
                                
            else:
                self.state=self.Kalman_1D(self.state,x_dot,process_var=0.05**2,measurement_var=0.1**2)
                self.filtered_buffer.append(self.threshold(self.state[0],0.2))
            self.TrendFilt(sh_buf_ln=5,filter_num=2,double_filt_flag=True)
            self.TrendFilt(sh_buf_ln=5,filter_num=2,double_filt_flag=False)
            return self.speeds[-1]
            
            # try:
            #     self.chart.compare_plot(self.buffer[-15:],self.filtered_buffer[-15:],self.Trend_buffer[-15:],self.speeds[-15:])
            #     return self.speeds[-1]
            # except:
            #     self.chart.compare_plot(self.buffer,self.filtered_buffer,self.Trend_buffer,self.speeds)
            #     return self.speeds[-1]

        else:

            if Kalman_flag:

                if first_run:
                    self.state=[x_dot,0.15**2]
                    self.filtered_buffer.append(x_dot)
                    print(self.filtered_buffer[-1])
                                
                else:
                    self.state=self.Kalman_1D(self.state,x_dot,process_var=0.05**2,measurement_var=0.1**2)
                    self.filtered_buffer.append(self.round_nearest(self.state[0],0.2))
                    print(self.filtered_buffer[-1])
                if double_filt_flag:    
                    self.TrendFilt(sh_buf_ln=5,filter_num=2,double_filt_flag=True)
                    # try:
                    #     self.chart.plot_data(self.speeds[-15:])
                    #     return self.speeds[-1]
                    # except:
                    #     self.chart.plot_data(self.speeds)
                    #     return self.speeds[-1]
                else:
                    pass
                    # try:
                    #     self.chart.plot_data(self.filtered_buffer[-15:])
                    #     return self.filtered_buffer[-1]
                    # except:
                    #     self.chart.plot_data(self.filtered_buffer)
                    #     return self.filtered_buffer[-1]
            if trend_flag:
                xx=self.TrendFilt(sh_buf_ln=5,filter_num=2,double_filt_flag=False)
                return xx
                # try:
                #     self.chart.plot_data(self.Trend_buffer[-15:],self.parameter[-15:])
                #     return self.Trend_buffer[-1]
                # except:
                #     self.chart.plot_data(self.Trend_buffer,self.parameter)
                #     return self.Trend_buffer[-1]
            
            if UKF_flag:
                
                self.TrendFilt(sh_buf_ln=5,filter_num=2,double_filt_flag=False)
                if len(self.buffer):
                    self.x_cov=0.8
                    self.cc=self.UKF(self.c,self.Qt,self.Rt,x_dot,self.h,n=1)
                else:
                    self.cc=self.UKF(self.c,self.Qt,self.Rt,x_dot,self.h,n=1)

                # try:
                #     self.chart.plot_data(self.Trend_buffer[-15:],self.parameter[-15:])
                #     return self.Trend_buffer[-1]
                # except:
                #     self.chart.plot_data(self.Trend_buffer,self.parameter)
                #     return self.Trend_buffer[-1]
            
            elif UKF_flag:
                if first_run:
                    x_cov=1
                    x_dot_prev=x_dot
                    self.cc=self.UKF(self.c,self.Qt,self.Rt,x_dot,self.h,self.delta_t,n=1)
                else:
                    self.cc=self.UKF(self.c,self.Qt,self.Rt,x_dot,self.h,self.delta_t,n=1)

            else:
                pass
            return self.filtered_buffer[-1]
    

"""

State space variables : Delta Y, delta t
Output : Y = Speed (measured)
Model parameter : C (This needs to be updated)

"""

class SpeedRecProcessing(SpeedRec): 
    def __init__(self, data_length=800, sample_rate=200, h=110, c=0.25):
        super().__init__(data_length, sample_rate, h, c)
        self.delta_ys=[]
        self.delta_xs=[]
        self.delta_ts=[]
        self.positions=[]
        self.velocties=[]
        self.calculated_speed=[]
        self.imu_data=[]
        self.split_data=[]
        



    def read_xdf(self,file_path):
        data,header=xdf.load_xdf(file_path)
        for data_series in data:
            if data_series['info']['name'][0]=='Speed_stream':
                self.calculated_speed=data_series['time_series']
            if data_series['info']['name'][0]=='polar accel':
                self.imu_data=data_series['time_series']
            else:
                pass
    
    def split_acc_data(self,imu_data,sample_rate=200,time_interval=4,split_interval=0.5):
        acc_data=[]
        i=0
        while (split_interval*i+time_interval)*sample_rate<len(imu_data):
           
            acc_data.append(imu_data[int(sample_rate*split_interval*i):int((split_interval*i+time_interval)*sample_rate)])
            i+=1
        return acc_data

    def get_deltas(self,data,delta_y=True,delta_t=True,delta_x=True,position=True,velocities=False):
        self.Velocity(data)
        deltas=[]
        if delta_y:
            self.delta_ys.append(self.delta_y)
            deltas.append(self.delta_y)
        if delta_t:
            self.delta_ts.append(self.delta_t)
            deltas.append(self.delta_t)
        if delta_x:
            self.delta_xs.append(self.delta_x)
            deltas.append(self.delta_x)
        if position:
            self.positions.append(self.position)
            deltas.append(self.position)
        if velocities:
            self.velocties.append(self.velocity)
            deltas.append(self.velocity)
        return deltas

    def Speed_estimation_simulation(self,filepath='',data=[],data_flag=False,s_rate=200,time_interval=4,split_interval=0.5,comapre_mode=True,Kalman=True,Trend=True,double_filt=True,sh_buffer_ln=5,filter_num=2,process_var=0.05**2,measurement_var=0.1**2,r_number=0.05,plotting=False):
            
        if not data_flag:
            self.read_xdf(filepath)
            self.split_data=self.split_acc_data(self.imu_data[:,0],s_rate,time_interval,split_interval)
            first_run=True
            
            if comapre_mode:
                first_data=self.split_data.pop(0)
                if first_run:
                    x_dot=self.Velocity(first_data,r_number)
                    self.state=[x_dot,0.25**2]
                    self.filtered_buffer.append(x_dot)
                    first_run=False
                
                for data in self.split_data:
                    x_dot=self.Velocity(data,r_number)
                    self.state=self.Kalman_1D(self.state,x_dot,process_var,measurement_var)
                    self.filtered_buffer.append(self.round_nearest(self.state[0],r_number))
                    self.TrendFilt(sh_buffer_ln,filter_num,False,r_number)
                    self.TrendFilt(sh_buffer_ln,filter_num,True,r_number)
                    
                    
                values={'Unfiltered':self.buffer,'Kalman':self.filtered_buffer,'Trend':self.Trend_buffer,'Double':self.speeds}
                return values
            else:
                pass
            
        else:
            self.split_data=self.split_acc_data(data,s_rate,time_interval,split_interval)

            first_run=True
            
            if comapre_mode:
                first_data=self.split_data.pop(0)
                if first_run:
                    x_dot=self.Velocity(first_data,r_number)
                    self.state=[x_dot,0.25**2]
                    self.filtered_buffer.append(x_dot)
                    first_run=False
                
                for data in self.split_data:
                    x_dot=self.Velocity(data,r_number)
                    self.state=self.Kalman_1D(self.state,x_dot,process_var,measurement_var)
                    self.filtered_buffer.append(self.round_nearest(self.state[0],r_number))
                    self.TrendFilt(sh_buffer_ln,filter_num,False,r_number)
                    self.TrendFilt(sh_buffer_ln,filter_num,True,r_number)
                    
                    
                values={'Unfiltered':self.buffer,'Kalman':self.filtered_buffer,'Trend':self.Trend_buffer,'Double':self.speeds}
                return values
            else:
                pass

    def C_search():
        pass
