import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
import heapq
import itertools
from pybullet_tools.utils import wait_for_user
from utils import draw_line

#########################

import itertools
def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []

    start_time = time.time()
    ### YOUR CODE HERE ###
    
    # grid details
    xlimit = 4.1 # x = [-xlimit, xlimit]
    ylimit = 2.1 # y = [-ylimit, ylimit]
    ang_res = np.pi/16
    lin_res = 0.05

    xrange = np.arange(-xlimit, xlimit+lin_res, lin_res)
    yrange = np.arange(-ylimit, ylimit+lin_res, lin_res)
    rrange = np.arange(0, 2*np.pi, ang_res)
    numx = len(xrange)
    numy = len(yrange)
    numr = len(rrange)

    start_config=np.array(start_config)
    goal_config=np.array(goal_config)
    start_config[2] %= 2*np.pi
    goal_config[2] %= 2*np.pi

    import matplotlib.pyplot as plt
    def viz_pts(pts):
        for p in pts:
            draw_sphere_marker((p[0], p[1], 1), 0.1, (1, 0, 0, 1))
    def viz_heatmap(pts):
        xidx = np.rint((pts[:,0] + xlimit)/lin_res).astype(np.int64)
        yidx = np.rint((pts[:,1] + ylimit)/lin_res).astype(np.int64)
        grid = np.ones((numx, numy), dtype=int)
        grid[xidx, yidx] = 0

        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.show()
        return

    def fill_random(q_rand):
        q_rand[:,0] = np.random.uniform(low=-xlimit, high=xlimit, size=rand_len)
        q_rand[:,1] = np.random.uniform(low=-ylimit, high=ylimit, size=rand_len)
        # q_rand[:,2] = np.random.uniform(low=0, high=2*np.pi, size=rand_len)

    num_nodes = 0
    tree_len = 16
    init = np.zeros(tree_len, dtype=bool)
    coords = np.zeros((tree_len, 3), dtype=np.float64) # you should exponentially resize this array
    coords[:] = np.inf
    nbrs_of = {} # maps each index to its nbrs
    ang_res = 0.05 # in rad
    goal_bias = 0.1
    step_size = 0.1
    # visualize RRT using python heatmap

    cur_rand = 0
    rand_len = 1024
    q_rand = np.zeros((rand_len, 3), dtype=np.float64)
    fill_random(q_rand)

    # init[0] = True
    coords[0] = [start_config[0], start_config[1], 0]
    # coords[0] = [0,0,0]
    nbrs_of[0] = []
    num_nodes = 1


    # puts the point at height=1.5 so it's better visible
    get_high = lambda s : (s[0], s[1], 1.5)
    goal_reached = False
    
    rand_idx = 0
    while (not goal_reached):
        if rand_idx >= rand_len:
            # refill rand array
            fill_random(q_rand)
            rand_idx = 0

        import random
        cur_rand = q_rand[rand_idx] if random.random() >= goal_bias else goal_config
        cur_rand[2] = 0
        # cur_rand = np.array([1,0,0])

        dists_sq = np.sum((coords[:num_nodes] - cur_rand)**2, axis=1)**(1/2)
        min_idx = int(np.argmin(dists_sq))
        cur_near = coords[min_idx]
        uvec = (cur_rand - cur_near)/dists_sq[min_idx]

        # print(cur_rand, cur_near, uvec)
        # print("dists", dists_sq[0])
        # draw_sphere_marker(get_high(cur_near), 0.1, (0, 1, 0, 1))
        # draw_sphere_marker(get_high(cur_rand), 0.1, (0, 1, 0, 1))
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
            # init[cur_idx] = True
            nbrs_of[prev_idx].append(cur_idx)
            nbrs_of[cur_idx] = [prev_idx]

            prev_idx = cur_idx
            cur_idx += 1
            # draw_sphere_marker(get_high(pt), 0.1, (1, 0, 0, 1))

        rand_idx += 1
        # if (rand_idx == 100):
        #     wait_for_user()
        # from pprint import pprint
        # print("Done. Press enter to continue")
        # print("coords\n", len(coords))
        # pprint(coords)
        # print("init\n", len(init))
        # pprint(init)
        # print("nbrs_of\n", len(nbrs_of))
        # pprint(nbrs_of)
        # wait_for_user()
        # exit(0)
        # q_near = np.where()


    # construct path
    cur = num_nodes - 1
    path=[]
    while True:
        path.append(coords[cur])
        cur = nbrs_of[cur][0]
        if cur == 0:
            break
    path = path[::-1]
    # from pprint import pprint
    # pprint(path[::-1])

    # draw tree edges. WARNING: Destructive: destroys nbrs_of
    for lidx in range(num_nodes):
        if lidx not in nbrs_of:
            continue
        for ridx in nbrs_of[lidx]:
            line_start = get_high(coords[lidx])
            line_end = get_high(coords[ridx])
            line_width = 1
            line_color = (1, 0, 0) # R, G, B

            if ridx not in nbrs_of:
                continue
            nbrs_of[ridx].remove(lidx)
            if not nbrs_of[ridx]:
                nbrs_of.pop(ridx)

            draw_line(line_start, line_end, line_width, line_color)
    # wait_for_user()





    # print("start", start_config)
    # print("idns", (xind, yind))
    # print("ind pos", to_coord(xind, yind))
    # print(start_config)

    # gp = lambda x : (-3.4+x,-1.4,0.0)
    # path = [gp(0.5*t) for t in range(10)]
    
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
