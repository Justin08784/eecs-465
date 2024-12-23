import numpy as np

###YOUR CODE HERE###
#motion model
A = np.eye(2) #replace with your A
B = np.array([
    [1.5, 0.1],
    [0.2, -0.5]
]) #replace with your B

#sensor model
C = np.array([
    [1.05, 0.01],
    [0.01, 0.90]
]) #replace with your C
    
#motion noise covariance
# NOTE: R.npy and Q.npy were created by "np.save"-ing the motion_cov
# and sensor_cov arrays respectively in tuning.py
R = np.load("R.npy") #replace with your R

#sensor noise covariance
Q = np.load("Q.npy") #replace with your Q
###YOUR CODE HERE###
