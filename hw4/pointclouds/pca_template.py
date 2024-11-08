#!/usr/bin/env python
import utils
import numpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import scipy as sp
import numpy as np

###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    orig_pc = np.array(pc)
    pc = orig_pc.copy()
    X = pc[:,:,0]
    m = pc.shape[1] # point dimension
    n = pc.shape[0] # number of points
    # Show the input point cloud
    fig1 = utils.view_pc([orig_pc])

    #Rotate the points to align with the XY plane
    mean = np.mean(X, axis=0)
    X -= mean # center to mean

    Y = X.T / ((n-1)**0.5)
    U, S, V_T = np.linalg.svd(Y)

    var = S**2
    # NOTE: Increase threshold or not?
    threshold = 1e-2
    # threshold = 4e-2
    keep = var >= threshold

    # BUG: The slide say to use V_T, but if I do that
    # then the points vanish (i.e. they all go to (0,0,0)).
    # How is that using U instead of V_T works?
    # Also, isn't Y a 3x200 matrix, meaning its V_T is a 200x200
    # matrix? How do you extract ≤3 cols from a 200x200 V_T?
    # NOTE: 3.a) rotation only
    # pc[:,:,0] = X @ U
    # NOTE: 3.b) rotation + dimensionality reduction
    pc[:,keep,0] = X @ U[:,keep]
    pc[:,~keep,0] = 0
    utils.view_pc([pc], fig=fig1, color='r')

    # BUG: Converting to matrix is necessary so that
    # d = -pt.T * normal is interpreted as matrix multiplication
    # instead of row-wise multiplication. A better solution is
    # to use numpy arrays instead and change to d = -pt.T @ normal
    nv = np.matrix(U[-1][:,None])
    centroid = np.matrix(np.mean(orig_pc, axis=0))
    utils.draw_plane(fig1, nv, centroid, color=(0, 0.4, 0, 0.3))

    # BUG: Why does the transformed pc look "stretched"?

    #Show the resulting point cloud


    #Rotate the points to align with the XY plane AND eliminate the noise


    # Show the resulting point cloud

    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
