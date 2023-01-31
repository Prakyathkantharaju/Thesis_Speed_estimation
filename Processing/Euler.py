import numpy as np
import matplotlib.pyplot as plt
import imufusion
import pyxdf as xdf
from Split import split
"""
Have a script here that will get the data in numpy arrays
Let's say we have the data in the following variables:
    - acc: numpy array of shape (N, 3) with the accelerometer data
    - gyr: numpy array of shape (N, 3) with the gyroscope data
    - mag: numpy array of shape (N, 3) with the magnetometer data
    - dt: numpy array of shape (N,) with the time differences between samples
"""

# Load the data
def get_euler(data):
    #data=np.empty()
    ahrs=imufusion.Ahrs()

    sample_rate=200

    # Run the filter

    n_samples=len(data)
    linear_acceleration=np.empty((n_samples,3))
    euler_angles=np.empty((n_samples,3))

    for i in range(n_samples):
        ahrs.update_no_magnetometer(data[i,:3],data[i,3:6],1/sample_rate)
        linear_acceleration[i,:]=ahrs.get_linear_acceleration()
        euler_angles[i,:]=ahrs.get_euler_angles()

    # Plot the results
    return linear_acceleration,euler_angles




