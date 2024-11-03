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
    pc = np.array(pc)
    X = pc[:,:,0]
    m = pc.shape[1] # point dimension
    n = pc.shape[0] # number of points
    # Show the input point cloud
    fig = utils.view_pc([pc])

    #Rotate the points to align with the XY plane
    mean = np.mean(X, axis=0)
    X -= mean # center to mean

    Y = X.T / ((n-1)**0.5)
    U, S, V_T = sp.linalg.svd(Y)

    var = S**2
    threshold = 1e-2
    keep = var >= threshold

    # BUG: The slide say to use V_T, but if I do that
    # then the points vanish (i.e. they all go to (0,0,0)).
    # How is that using U instead of V_T works?
    # NOTE: 3.a) rotation only
    # pc[:,:,0] = X @ U
    # NOTE: 3.b) rotation + dimensionality reduction
    pc[:,keep,0] = X @ U[:,keep]
    pc[:,~keep,0] = 0
    fig = utils.view_pc([pc])

    # BUG: Converting to matrix is necessary so that
    # d = -pt.T * normal is interpreted as matrix multiplication
    # instead of row-wise multiplication. A better solution is
    # to use numpy arrays instead and change to d = -pt.T @ normal
    nv = np.matrix([np.ones(m) * ~keep]).T
    pt = np.matrix([[0],[0],[0]])
    # WARNING: I think this is wrong. We should draw a plane for the
    # ORIGINAL pc?
    utils.draw_plane(fig, nv, pt)


    #Show the resulting point cloud


    #Rotate the points to align with the XY plane AND eliminate the noise


    # Show the resulting point cloud

    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
