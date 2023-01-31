import numpy as np
import os
import matplotlib.pyplot as plt
import pyxdf as xdf


def split(data,length=400,interval=0.2,sample_rate=200):
    """
    Split the data into chunks of length samples with an interval of interval seconds between them.
    """
    n_samples=len(data)
    n_chunks=int((n_samples-length)/(interval*sample_rate))
    chunks=np.empty((n_chunks,length,6))
    for i in range(n_chunks):
        chunks[i,:,:]=data[i*int(interval*sample_rate):i*int(interval*sample_rate)+length,:]
    return chunks
