import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###
import heapq
import itertools
from pybullet_tools.utils import wait_for_user
from utils import draw_line
import math
import time
import random

#########################


joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit, get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}

    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))

    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###
    # grid details
    start = time.time()
    start_config=np.array(start_config)
    goal_config=np.array(goal_config)

    jnt_limits = []
    for i in range(6):
        name = joint_names[i]
        lims = joint_limits[name]
        jnt_limits.append([lims[0], lims[1]])
    jnt_limits = np.array(jnt_limits)

    def fill_random(q_rand):
        for i in range(6):
            q_rand[:,i] = np.random.uniform(low=jnt_limits[i,0], high=jnt_limits[i,1], size=q_rand.shape[0])

    num_nodes = 0
    tree_len = 16
    coords = np.zeros((tree_len, 6), dtype=np.float64) # exponentially resized
    coords[:] = np.inf
    nbrs_of = {} # maps each index to its nbrs (first nbr is always parent)
    step_size = 0.05
    goal_bias = 0.1

    rand_len = 1024
    q_rand = np.zeros((rand_len, 6), dtype=np.float64)
    fill_random(q_rand)

    # add root (start)
    coords[0] = start_config
    nbrs_of[0] = []
    num_nodes = 1
    goal_reached = False
    rand_idx = 0
    while (not goal_reached):
        if rand_idx >= rand_len:
            # refill rand array
            fill_random(q_rand)
            rand_idx = 0

        cur_rand = q_rand[rand_idx] if random.random() >= goal_bias else goal_config
        cur_rand[2] = 0

        dists_sq = np.minimum(
            np.sum((coords[:num_nodes] - cur_rand)**2, axis=1),
            np.sum((2*np.pi - abs(coords[:num_nodes] - cur_rand))**2, axis=1)
        )**0.5

        min_idx = int(np.argmin(dists_sq))
        cur_near = coords[min_idx]
        uvec = (cur_rand - cur_near)/dists_sq[min_idx]

        max_step = int(dists_sq[min_idx]/step_size)
        prev_idx = min_idx
        cur_idx = num_nodes

        for t in range(1, max_step + 1):
            pt = cur_near+t*step_size*uvec
            if (collision_fn(pt)):
                break
            goal_reached = np.sum((pt - goal_config)**2)**(0.5) < step_size

            if (num_nodes >= tree_len):
                # resize tree array
                new_arr = np.zeros((coords.shape[0] * 2, coords.shape[1]))
                new_arr[:num_nodes, :] = coords
                coords = new_arr
                tree_len *= 2

            num_nodes += 1

            coords[cur_idx] = pt
            nbrs_of[prev_idx].append(cur_idx)
            nbrs_of[cur_idx] = [prev_idx]

            prev_idx = cur_idx
            cur_idx += 1

        rand_idx += 1


    # construct path
    cur = num_nodes - 1
    raw_path=[]
    while True:
        raw_path.append(coords[cur])
        # parent of cur; we can do this thanks to topological ordering
        cur = nbrs_of[cur][0]
        if cur == 0:
            break
    raw_path = np.array(raw_path)
    raw_path = raw_path[::-1]
    print("runtime: ", time.time() - start)

    num_iters = 150
    params = np.sort(np.random.uniform(low=0,high=1.0,size=(num_iters,2)), axis=1)


    q_dim = 6
    raw_num_nodes = raw_path.shape[0]
    num_nodes = raw_num_nodes
    cur = np.zeros(shape=(raw_num_nodes, q_dim+1))
    cur[:,:q_dim] = raw_path
    cur[1:,q_dim] = np.sum((cur[1:,:q_dim] - cur[:-1,:q_dim])**2, axis=1)**(1/2) # dists col
    nex = np.zeros(shape=(raw_num_nodes, q_dim+1))

    # all the edges should be of length step_size=0.05
    assert(np.allclose(cur[1:,q_dim], np.asarray(step_size)))

    for i in range(num_iters):
        cumsums = cur[:num_nodes,q_dim].cumsum()
        path_len = cumsums[num_nodes-1]
        llen, rlen = params[i] * path_len
        # cumsums[lidx-1] <= llen < cumsums[lidx]
        # node_{lidx-1} <= left_endpoint < node_{lidx}
        lidx = np.searchsorted(cumsums[:num_nodes], llen, side='right')
        assert(lidx > 0)
        # cumsums[ridx-1] <= llen < cumsums[ridx]
        # node_{ridx-1} <= right_endpoint < node_{ridx}
        ridx = lidx + np.searchsorted(cumsums[lidx:num_nodes], rlen, side='right')
        assert(ridx >= lidx)
        if lidx == ridx:
            # same edge, just skip
            continue

        ledge_len = cur[lidx, q_dim]
        redge_len = cur[ridx, q_dim]
        lt = (llen - cumsums[lidx-1]) / ledge_len
        rt = (rlen - cumsums[ridx-1]) / redge_len

        # TODO: Try debugging this by visualizing the arm config points
        # i.e. draw a sphere for each endpoint and green edge (on top of the original path)
        lq = cur[lidx-1,:q_dim] + lt*(cur[lidx,:q_dim] - cur[lidx-1,:q_dim])
        rq = cur[ridx-1,:q_dim] + rt*(cur[ridx,:q_dim] - cur[ridx-1,:q_dim])
        
        # way too close to endpoint nodes; float error my throw some shit
        if np.allclose(lq, cur[lidx-1,:q_dim]) or np.allclose(lq, cur[lidx,:q_dim]):
            print("warning: too close left")
            # assert False, "too close left"
            continue
        if np.allclose(rq, cur[ridx-1,:q_dim]) or np.allclose(rq, cur[ridx,:q_dim]):
            print("warning: too close right")
            # assert False, "too close right"
            continue

        vec_norm = np.sum((rq - lq)**2)**0.5
        uvec = (rq - lq) / vec_norm

        max_step = int(vec_norm/step_size)
        collides=False
        for t in range(1, max_step + 1):
            q = lq + t*step_size*uvec
            if (collision_fn(q)):
                collides=True
                break
        if collides:
            continue

        delta = lidx - ridx + 2
        nex_num_nodes = num_nodes + delta
        if nex_num_nodes > raw_num_nodes:
            print("warning: raw_num_nodes exceeded")
            continue

        nex[:lidx] = cur[:lidx]
        nex[lidx+2:nex_num_nodes] = cur[ridx:num_nodes]

        nex[lidx,:q_dim] = lq
        nex[lidx,q_dim] = llen - cumsums[lidx-1]

        nex[lidx+1,:q_dim] = rq
        nex[lidx+1,q_dim] = vec_norm

        nex[lidx+2, q_dim] = np.sum((nex[lidx+2,:q_dim] - nex[lidx+1,:q_dim])**2)**(1/2)

        num_nodes = nex_num_nodes
        tmp = cur
        cur = nex
        nex = tmp
        continue

    smoothed_path = cur[:num_nodes,:q_dim]


    def get_ee_positions(path):
        PR2 = robots['pr2']
        path_positions = []
        for st in path:
            set_joint_positions(PR2, joint_idx, st)
            path_positions.append(get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))[0])
        return path_positions
    def draw_path(path, col_code=(1,0,0), line_width=2):
        for i in range(len(path) - 1):
            p_i = path[i]
            p_f = path[i+1]
            draw_line(p_i, p_f, line_width, col_code)

    draw_path(get_ee_positions(raw_path), col_code=(1,0,0))
    draw_path(get_ee_positions(smoothed_path), col_code=(0,1,0), line_width=4)
    path = smoothed_path


    # WARNING: You tried your best, but the following piece of code is no good.
    # It linearly interplates in the CONFIGURATION space, not the task space.
    # What is linear in the configuration space is often curved in the task space,
    # leading to deviation from the true smoothed path.

    # new_path = [cur[0,:q_dim]]
    # for i in range(1, num_points):
    #     if cur[i,q_dim] < step_size:
    #         new_path.append(cur[i,:q_dim])
    #         continue
    #     max_steps = int(cur[i,q_dim]//step_size)
    #     vec = cur[i,:q_dim] - cur[i-1,:q_dim]
    #     for t in range(1, max_steps + 1):
    #         x = cur[i-1,:q_dim] + vec * t/max_steps
    #         new_path.append(x)
    #     new_path.append(cur[i,:q_dim])
    # new_path = np.array(new_path)
    # smoothed_path = new_path









    # print("start", start_config)
    # print("idns", (xind, yind))
    # print("ind pos", to_coord(xind, yind))
    # print(start_config)

    

    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
