#!/usr/bin/env python
import utils
import numpy
import time
import random
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


delta = 0.1
def ransac(pc, iter_limit=5000):
    pc = np.array(pc)[:,:,0]
    n = pc.shape[0]
    m = pc.shape[1]

    # NOTE: Fixed seed! Disable later?
    seed = 0 # fix seed
    rng = np.random.default_rng(seed)
    choices = np.array([rng.choice(n, m, replace=False) for _ in range(iter_limit)])

    in_sample = np.zeros(shape=n, dtype=bool)
    in_consensus = np.zeros(shape=n, dtype=bool)
    min_num_consensus = 150
    best_error = np.inf
    best_nv = np.zeros(m, dtype=np.float64)
    best_off = 0
    for i in range(iter_limit):
        # print(best_error)
        in_sample[choices[i]] = True
        cur = pc[in_sample,:]
        # print("choices", cur)

        # fig, ax = create_plot()
        # draw_pc(ax, pc[~in_sample], color='b', alpha=0.1)
        # draw_pc(ax, pc[in_sample], color='r', alpha=1)



        v1 = cur[1] - cur[0]
        v2 = cur[2] - cur[1]
        nv = np.cross(v1, v2)
        nv /= np.linalg.norm(nv)
        off = -np.dot(nv, cur[0])


        # draw_plane(fig, nv, cur[0])
        # draw_plane(fig, nv, cur[0] + delta * nv)
        # draw_plane(fig, nv, cur[0] - delta * nv)


        errors = np.abs(np.dot(pc, nv) + off)

        nums = np.arange(n)
        in_consensus = (errors < delta) & ~in_sample
        num_consensus = np.count_nonzero(in_consensus)

        # draw_pc(ax, pc[in_consensus], color='g', alpha=0.3)
        if num_consensus < min_num_consensus:
            in_consensus[:] = False
            in_sample[:] = False
            continue

        in_both = in_consensus | in_sample
        num_both = num_consensus + m
        assert(num_both == np.count_nonzero(in_both))
        A = np.zeros((num_both, m), dtype=np.float64)
        B = pc[in_both,m-1]
        A[:,:m-1] = pc[in_both,:m-1]
        A[:,m-1] = 1

        A_pinv = np.linalg.pinv(A)
        a, b, off = A_pinv @ B
        nv = np.array([a, b, -1])
        nv_norm = np.linalg.norm(nv)
        # normalize plane parameters
        nv /= nv_norm
        off /= nv_norm
        error = np.mean(np.abs(np.dot(pc[in_both], nv) + off))

        # draw_plane(fig, nv, cur[0], color=(0, 0.5, 0, 0.3))

        # plt.show()
        # exit(0)
        if error < best_error:
            best_nv[:] = nv
            best_off = off
            best_error = error

        in_consensus[:] = False
        in_sample[:] = False


    fig, ax = create_plot()
    d = -best_off / best_nv[2]
    print("best_nv", best_nv)
    print("d", d)
    pt = np.array([0,0,d])

    errors = np.abs(np.dot(pc, best_nv) + best_off)
    inliers = errors < delta

    draw_pc(ax, pc[~inliers], color='b', alpha=0.2)
    draw_pc(ax, pc[inliers], color='r', alpha=0.5)
    draw_plane(fig, best_nv, pt, color=(0.0, 0.4, 0.0, 0.3))

def pca(pc):
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
    # utils.view_pc([pc], fig=fig1, color='r')

    # BUG: Converting to matrix is necessary so that
    # d = -pt.T * normal is interpreted as matrix multiplication
    # instead of row-wise multiplication. A better solution is
    # to use numpy arrays instead and change to d = -pt.T @ normal
    # nv = np.matrix(U[-1][:,None])
    orig_pc = np.array(orig_pc)[:,:,0]
    nv = U[-1]
    centroid = np.mean(orig_pc, axis=0)
    # draw_plane(fig1, nv, centroid, color=(0, 0.4, 0, 0.3))

    # BUG: Why does the transformed pc look "stretched"?
    off = -np.dot(nv.T, centroid.T)
    errors = np.abs(np.dot(orig_pc, nv) + off)
    inliers = errors < delta

    fig, ax = create_plot()
    draw_pc(ax, orig_pc[~inliers], color='b', alpha=0.2)
    draw_pc(ax, orig_pc[inliers], color='r', alpha=0.5)
    draw_plane(fig, nv, centroid, color=(0.0, 0.4, 0.0, 0.3))

###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    fig = None
    for i in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        fig = utils.view_pc([pc])

        # pca(pc)
        # plt.show()
        # exit(0)

        ###YOUR CODE HERE###


        #this code is just for viewing, you can remove or change it
        input("Press enter for next test:")
        plt.close(fig)
        ###YOUR CODE HERE###

    input("Press enter to end")


if __name__ == '__main__':
    main()
