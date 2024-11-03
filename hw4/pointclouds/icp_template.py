#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np

def create_plot():
    plt.ion()
    # Make a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig, ax

def draw_pc(ax, pc, color, marker='o', alpha=0.5):
    ax.scatter3D(pc[:,0], pc[:,1], pc[:,2], color=color, marker=marker, alpha=alpha)
    plt.draw()
    plt.pause(0.05)
    plt.ioff() #turn off interactive plotting


def draw_plane(fig, normal, pt, color=(0.1, 0.2, 0.5, 0.3), length=[-1, 1], width=[-1, 1]):
    # Calculate d in ax + by + cz + d = 0
    d = -np.dot(pt, normal)

    # Calculate points on the surface
    x = 0
    y = 0
    z = 0
    if normal[2] != 0:
        x, y = numpy.meshgrid(numpy.linspace(length[0], length[1], 10),
                              numpy.linspace(width[0], width[1], 10))
        z = (-d - normal[0] * x - normal[1] * y) / normal[2]
    elif normal[1] != 0:
        x, z = numpy.meshgrid(numpy.linspace(length[0], length[1], 10),
                              numpy.linspace(width[0], width[1], 10))
        y = (-d - normal[0] * x - normal[2] * z) / normal[1]
    elif normal[0] != 0:
        y, z = numpy.meshgrid(numpy.linspace(length[0], length[1], 10),
                              numpy.linspace(width[0], width[1], 10))
        x = (-d - normal[1] * y - normal[2] * z) / normal[0]

    # Plot the surface
    ax = fig.gca()
    ax.plot_surface(x, y, z, color=color)
    # Update the figure
    plt.draw()
    plt.pause(0.05)
    plt.ioff() #turn off interactive plotting
    return fig

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target


    pc_source = np.array(pc_source)[:,:,0]
    pc_target = np.array(pc_target)[:,:,0]


    from pprint import pprint
    pprint(pc_source)
    fig, ax = create_plot()
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    # centralized clouds
    src_mean = np.mean(pc_source, axis=0)
    dst_mean = np.mean(pc_target, axis=0)
    src = pc_source - src_mean
    dst = pc_target - dst_mean

    S = src.T @ dst
    U, S, V_T = np.linalg.svd(S)
    V = V_T.T
    det = np.linalg.det(V @ U.T)
    print(U.shape)
    print(S.shape)
    print(V_T.shape)
    thismatrix = np.identity(3)
    thismatrix[2,2] = det
    R = V @ thismatrix @ U.T
    t = dst_mean - R @ src_mean


    nahman = (R @ pc_source.T).T + t




    print(R)
    print(t)

    draw_pc(ax, pc_source, color='b', marker='o')
    draw_pc(ax, nahman, color='g', marker='o')
    draw_pc(ax, pc_target, color='r', marker='^')
    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
