#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

if __name__ == '__main__':
    
    #initialize plotting        
    plt.ion()
    
    #load in the data
    PIK = "kfdata.dat"
    with open(PIK, "rb") as f:
        noisy_measurement,actions,ground_truth_states,N = pickle.load(f)

    #your model parameters are imported here
    from kfmodel import A, B, C, Q, R

    #we are assuming both the motion and sensor noise is 0 mean
    motion_errors = np.zeros((2,N))
    sensor_errors = np.zeros((2,N))
    for i in range(1,N):
        x_t = np.matrix(ground_truth_states[:,i]).transpose()
        x_tminus1 = np.matrix(ground_truth_states[:,i-1]).transpose()
        u_t = np.matrix(actions[:,i]).transpose()
        z_t = np.matrix(noisy_measurement[:,i]).transpose()
        ###YOUR CODE HERE###
        motion_errors[:,1:] = ground_truth_states[:,1:] - A @ ground_truth_states[:,:-1] - B @ actions[:,1:]
        sensor_errors[:,1:] = noisy_measurement[:,1:] - C @ ground_truth_states[:,1:]
        break
        ###YOUR CODE HERE###
    
    motion_cov=np.cov(motion_errors)
    sensor_cov=np.cov(sensor_errors)
    
    print("Motion Covariance:")
    print(motion_cov)
    print("Measurement Covariance:")
    print(sensor_cov)

    
