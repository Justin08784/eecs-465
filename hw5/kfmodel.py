import numpy as np

###YOUR CODE HERE###
#motion model
A = np.eye(2) #replace with your A
B = np.eye([
    [1.5, 0.1],
    [0.2, -0.5]
]) #replace with your B

#sensor model
C = np.eye([
    [1.05, 0.01],
    [0.01, 0.90]
]) #replace with your C
    
#motion noise covariance
R = np.eye(2) #replace with your R

#sensor noise covariance
Q = np.eye(2) #replace with your Q
###YOUR CODE HERE###
