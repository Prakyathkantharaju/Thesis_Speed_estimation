
from typing import List


import numpy as np
import math
import pylsl
import time


class Inlet:
    """Parent class
    """
    def __init__(self, info: pylsl.StreamInfo, data_length: int):
        # fill the basic information from the stream.
        self.inlet = pylsl.StreamInlet(info, max_buflen = data_length,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.new_data = False


class DataInlet(Inlet):
    """ Class to stream the data for all the channels"""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, data_length: int):
        super().__init__(info, data_length)
        buffer_size = (2 * math.ceil(info.nominal_srate() * data_length), info.channel_count())
        self.buffer = np.empty(buffer_size, dtype = self.dtypes[info.channel_format()])
        self.first_data = True
        self.store_data = []

    def start_data(self,n):
        # pull the data and store in the buffer
        _, ts = self.inlet.pull_chunk(timeout = 0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)

        # if ts is present then the data is present.
        if ts:
            ts = np.asarray(ts)
            if self.name  == 'polar accel':
                if self.first_data:
                    self.store_data = self.buffer[0:ts.size,:]
                    self.first_data = False
                else:
                    #print(self.store_data.shape, self.buffer[0:ts.size,:].shape)
                    self.store_data = np.append(self.store_data, self.buffer[0:ts.size], axis = 0)
                    if len(self.store_data) > n:
                        self.new_data = True


    def get_data(self, n= 100):
        self.start_data(n=n)
        if self.new_data:
            #print(len(self.store_data))
            send_data =  self.store_data[-n:,:]
            #print(send_data.shape)
            return send_data
        else:
            return np.zeros((100, 3))

class SetupStreams():
    def __init__(self):
        self.inlets: List[Inlet] = []
        self.streams = pylsl.resolve_streams()

        self.acc_index = 1
        i = 0
        for info in self.streams:
            data_inlet = DataInlet(info, 1000)
            
            print(info.name())
            if info.name() == 'polar accel':
                self.inlets.append(data_inlet)
                print(len(self.inlets))
                self.acc_index = i
            i += 1
        # self.acc_index = 0

    def get_data(self,n):
        return self.inlets[0].get_data(n)

    def run(self):
        for inlet in self.inlets:
            inlet.get_data()
