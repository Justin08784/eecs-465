import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###
import heapq
import itertools
from pybullet_tools.utils import wait_for_user
from utils import draw_line

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
    start_config=np.array(start_config)
    goal_config=np.array(goal_config)
    # TODO: Why is this not allowed? Is the range restricted to [-np.pi, np.pi]?
    # start_config[:] %= 2*np.pi
    # goal_config[:] %= 2*np.pi
    # BUG: If you do the 2*np.pi normalization, then start_config––and every pt in the main for-loop––
    # is in collision. e.g. try:
    # print("ding dong fuck", collision_fn(np.array(start_config)))
    # exit(0)

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
        q_rand[:] = np.random.uniform(low=-np.pi, high=np.pi, size=q_rand.shape)

    num_nodes = 0
    tree_len = 16
    coords = np.zeros((tree_len, 6), dtype=np.float64) # you should exponentially resize this array
    coords[:] = np.inf
    nbrs_of = {} # maps each index to its nbrs
    step_size = 0.05
    goal_bias = 0.1
    # visualize RRT using python heatmap

    cur_rand = 0
    rand_len = 1024
    q_rand = np.zeros((rand_len, 6), dtype=np.float64)
    fill_random(q_rand)

    coords[0] = start_config
    nbrs_of[0] = []
    num_nodes = 1


    # puts the point at height=1.5 so it's better visible
    goal_reached = False

    rand_idx = 0
    iter = 0
    while (not goal_reached):
        if rand_idx >= rand_len:
            # refill rand array
            fill_random(q_rand)
            rand_idx = 0

        import random
        cur_rand = q_rand[rand_idx] if random.random() >= goal_bias else goal_config
        cur_rand[2] = 0
        # cur_rand = np.array([1,0,0])

        dists_sq = np.minimum(
            np.sum((coords[:num_nodes] - cur_rand)**2, axis=1),
            np.sum((2*np.pi - abs(coords[:num_nodes] - cur_rand))**2, axis=1)
        )**0.5
        # np.minimum(
        #     abs(expandee_coords[2] - nbrs_coords[:, 2]),
        #     2*np.pi - abs(expandee_coords[2] - nbrs_coords[:, 2])
        # )**2
        min_idx = int(np.argmin(dists_sq))
        cur_near = coords[min_idx]
        uvec = (cur_rand - cur_near)/dists_sq[min_idx]
        # if iter % 1000 == 0:
        #     print(iter, dists_sq[min_idx])
        # iter+=1

        # print(cur_rand, cur_near, uvec)
        # print("dists", dists_sq[0])
        # draw_sphere_marker(get_high(cur_near), 0.1, (0, 1, 0, 1))
        # draw_sphere_marker(get_high(cur_rand), 0.1, (0, 1, 0, 1))
        max_step = int(dists_sq[min_idx]/step_size)
        prev_idx = min_idx
        cur_idx = num_nodes
        # print("vecs")
        # print(cur_near)
        # print(cur_rand)
        # print(uvec)
        # print("Default")
        # print(collision_fn(start_config))
        for t in range(1, max_step + 1):
            pt = cur_near+t*step_size*uvec
            if (collision_fn(pt)):
                break
            # print(collision_fn(pt))
            # assert(False)
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
        # parent of cur; we can do this thanks to topological ordering
        cur = nbrs_of[cur][0]
        if cur == 0:
            break
    path = path[::-1]
    # from pprint import pprint
    # pprint(path[::-1])

    # draw tree edges. WARNING: Destructive: destroys nbrs_of
    # for lidx in range(num_nodes):
    #     if lidx not in nbrs_of:
    #         continue
    #     for ridx in nbrs_of[lidx]:
    #         line_start = get_high(coords[lidx])
    #         line_end = get_high(coords[ridx])
    #         line_width = 1
    #         line_color = (1, 0, 0) # R, G, B

    #         if ridx not in nbrs_of:
    #             continue
    #         nbrs_of[ridx].remove(lidx)
    #         if not nbrs_of[ridx]:
    #             nbrs_of.pop(ridx)

    #         draw_line(line_start, line_end, line_width, line_color)
    # wait_for_user()





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
