#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np

def myplot(pc):
    plt.ion()
    # Make a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(pc[:,0], pc[:,1], pc[:,2], color='b', marker='o')

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Update the figure
    plt.draw()
    plt.pause(0.05)
    plt.ioff() #turn off interactive plotting
    plt.show()

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    # fig = utils.view_pc([pc])

    #Fit a plane to the data using ransac
    iter_limit = 1000
    pc = np.array(pc)[:,:,0]
    from pprint import pprint
    n = pc.shape[0]
    m = pc.shape[1]

    # NOTE: Fixed seed! Disable later?
    seed = 0 # fix seed
    rng = np.random.default_rng(seed)
    choices = np.array([rng.choice(n, m, replace=False) for _ in range(iter_limit)])

    delta = 0.5
    in_sample = np.zeros(shape=n, dtype=bool)
    in_consensus = np.zeros(shape=n, dtype=bool)
    min_num_consensus = 80
    best_error = np.inf
    best_nv = np.zeros(m, dtype=np.float64)
    best_off = 0
    for i in range(iter_limit):
        # print(best_error)
        in_sample[choices[i]] = True
        cur = pc[in_sample,:]
        print("choices", cur)
        v1 = cur[1] - cur[0]
        v2 = cur[2] - cur[1]
        nv = np.cross(v1, v2)
        nv /= np.linalg.norm(nv)
        off = -np.dot(nv, cur[0])

        errors = np.abs(np.dot(pc, nv) + off)

        nums = np.arange(n)
        in_consensus = (errors < delta) & ~in_sample
        num_consensus = np.count_nonzero(in_consensus)

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
        nv = np.array([a, b, 1])
        nv /= np.linalg.norm(nv)
        error = np.mean(np.abs(np.dot(pc[in_both], nv) + off))

        if error < best_error:
            best_nv[:] = nv
            best_off = off
            best_error = error

        in_consensus[:] = False
        in_sample[:] = False


    nv = np.matrix(nv).T
    pt = np.matrix([[0],[0],[0]])
    # utils.draw_plane(fig, nv, pt)



    #Show the resulting point cloud

    #Draw the fitted plane


    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
