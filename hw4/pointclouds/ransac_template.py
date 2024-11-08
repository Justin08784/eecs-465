#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np


###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
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

    # Show the input point cloud
    # fig = utils.view_pc([pc])

    #Fit a plane to the data using ransac
    iter_limit = 5000
    pc = np.array(pc)[:,:,0]
    from pprint import pprint
    n = pc.shape[0]
    m = pc.shape[1]

    # NOTE: Fixed seed! Disable later?
    seed = 0 # fix seed
    rng = np.random.default_rng(seed)
    choices = np.array([rng.choice(n, m, replace=False) for _ in range(iter_limit)])

    delta = 0.1
    in_sample = np.zeros(shape=n, dtype=bool)
    in_consensus = np.zeros(shape=n, dtype=bool)
    min_num_consensus = 200
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
        error = np.sum(np.abs(np.dot(pc[in_both], nv) + off))

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
    # draw_plane(fig, best_nv, pt + delta * best_nv, color=(0.1, 0.2, 0.5, 0.1))
    # draw_plane(fig, best_nv, pt - delta * best_nv, color=(0.1, 0.2, 0.5, 0.1))


    #Show the resulting point cloud

    #Draw the fitted plane


    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
