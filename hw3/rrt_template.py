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
    # WARNING: remember to disable this (or remove)
    use_precomputed_path = True
    # grid details
    import time
    start = time.time()
    start_config=np.array(start_config)
    goal_config=np.array(goal_config)
    # TODO: Why is this not allowed? Is the range restricted to [-np.pi, np.pi]?
    # start_config[:] %= 2*np.pi
    # goal_config[:] %= 2*np.pi
    # BUG: If you do the 2*np.pi normalization, then start_config––and every pt in the main for-loop––
    # is in collision. e.g. try:
    # print("ding dong fuck", collision_fn(np.array(start_config)))
    # exit(0)

    jnt_limits = []
    for i in range(6):
        name = joint_names[i]
        lims = joint_limits[name]
        jnt_limits.append([lims[0], lims[1]])
    jnt_limits = np.array(jnt_limits)

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
        for i in range(6):
            q_rand[:,i] = np.random.uniform(low=jnt_limits[i,0], high=jnt_limits[i,1], size=q_rand.shape[0])

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
    while (not goal_reached and not use_precomputed_path):
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
    if not use_precomputed_path:
        while True:
            path.append(coords[cur])
            # parent of cur; we can do this thanks to topological ordering
            cur = nbrs_of[cur][0]
            if cur == 0:
                break
        path = np.array(path)
        path = path[::-1]
    else:
        path=np.load("raw_path.npy")
    print("runtime: ", time.time() - start)

    #ee_pose[0] is the translation of the left gripper tool frame
    #ee_pose[1] is the rotation (represented as a quaternion the left gripper tool frame), we don't need this
    def get_ee_positions(path):
        PR2 = robots['pr2']
        path_positions = []
        for st in path:
            set_joint_positions(PR2, joint_idx, st)
            path_positions.append(get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))[0])
        return path_positions

    num_iters = 150

    params = np.sort(np.random.uniform(low=0,high=1.0,size=(num_iters,2)), axis=1)
    cur = np.zeros(shape=(path.shape[0], 6+1))
    valid = np.ones(path.shape[0], dtype=bool)
    cur[:,:6] = path
    # dists col
    cur[1:,6] = np.sum((cur[1:,:6] - cur[:-1,:6])**2, axis=1)**(1/2)
    # all the edges should be of length step_size=0.05
    assert(np.allclose(cur[1:,6], np.asarray(step_size)))
    path_len = np.sum(cur[:-1,6])
    for i in range(num_iters):
        tmp = cur[valid]
        tmp_path = path[valid]
        cumsums = tmp[:,6].cumsum()
        path_len = cumsums[-1]
        llen, rlen = params[i] * path_len
        # llen, rlen = 0.0001, 0.1501
        lidx = np.searchsorted(cumsums, llen, side='right')
        ridx = lidx + np.searchsorted(cumsums[lidx:], rlen, side='right')
        if lidx == ridx:
            # same edge, just skip
            continue

        lt = (llen - cumsums[lidx-1]) / cumsums[lidx]
        rt = (rlen - cumsums[ridx-1]) / cumsums[ridx]

        # TODO: Try debugging this by visualizing the arm config points
        # i.e. draw a sphere for each endpoint and green edge (on top of the original path)
        lq = tmp_path[lidx-1] + lt*(tmp_path[lidx] - tmp_path[lidx-1]) / cumsums[lidx]
        rq = tmp_path[ridx-1] + rt*(tmp_path[ridx] - tmp_path[ridx-1]) / cumsums[lidx]
        
        # way too close to endpoint nodes; float error my throw some shit
        if np.allclose(lq, tmp_path[lidx-1]) or np.allclose(lq, tmp_path[lidx]):
            print("too close left")
            continue
        if np.allclose(rq, tmp_path[ridx-1]) or np.allclose(rq, tmp_path[ridx]):
            print("too close right")
            continue

        vec_norm = np.sum((rq - lq)**2)**0.5
        uvec = (rq - lq) / vec_norm
        # print(abs(lq-path[lidx-1]), abs(rq-path[ridx-1]))
        # print(vec_norm)
        # print(uvec)

        max_step = int(vec_norm/step_size)
        collides=False
        for t in range(1, max_step + 1):
            q = lq + t*step_size*uvec
            if (collision_fn(q)):
                collides=True
                break
        if collides:
            print("collides")
            continue


        if ((lidx-1)+1 == ridx-1):
            # TODO: resize array +1 when nodes are in adjacent edges
            continue
            assert False, "need to handle this stupid case"
            print("shit")
        else:
            assert((lidx-1)+1 < ridx-1)
            tmp[lidx,:6] = lq
            tmp[lidx+1,:6] = rq
            valid[lidx+2:ridx-1]=False
            print(path[valid].shape)
            # print(lidx-1,ridx, cur[lidx-1:ridx])
            # print(lidx-1, ridx, valid[lidx-1:ridx])
        # lql, lqr = path[lidx,lidx+1]
        # rql, rqr = path[ridx,ridx+1]
        # print(lq, rq)
        print(f"({llen},{rlen})->({lidx},{ridx}), {lt}, {rt}")
        # if i==100:
        #     exit(0)

        pass

    # print(cur)
    # print(valid)
    # exit(0)
    # print(params)
    # BUG: smoothing is clearly not working. It clips through walls and shit.
    smoothed_path = path[valid]
    

    # np.save("raw_path.npy", path)
    raw_positions = get_ee_positions(path)
    for i in range(len(raw_positions) - 1):
        line_start = raw_positions[i]
        line_end = raw_positions[i+1]
        line_width = 1
        line_color = (1, 0, 0) # R, G, B
        draw_line(line_start, line_end, line_width, line_color)

    raw_positions = get_ee_positions(smoothed_path)
    for i in range(len(raw_positions) - 1):
        line_start = raw_positions[i]
        line_end = raw_positions[i+1]
        line_width = 3
        line_color = (0, 1, 0) # R, G, B
        draw_line(line_start, line_end, line_width, line_color)
    wait_for_user()
    path=smoothed_path

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
