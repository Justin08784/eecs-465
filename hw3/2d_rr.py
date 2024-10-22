import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
import heapq
import itertools
from pybullet_tools.utils import wait_for_user, remove_all_debug
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
    use_precomputed_path = True

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
    while (not goal_reached and not use_precomputed_path):
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

    def draw_path(path):
        for i in range(len(path) - 1):
            line_start = get_high(path[i])
            line_end = get_high(path[i+1])
            line_width = 1
            line_color = (1, 0, 0) # R, G, B
            draw_line(line_start, line_end, line_width, line_color)

    path = np.load("raw_2dpath.npy")
    path = np.array(path)

    def get_ee_positions(path):
        PR2 = robots['pr2']
        path_positions = []
        for st in path:
            set_joint_positions(PR2, joint_idx, st)
            path_positions.append(get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))[0])
        return path_positions

    num_iters = 150
    params = np.sort(np.random.uniform(low=0,high=1.0,size=(num_iters,2)), axis=1)

    q_dim = 3
    num_points = path.shape[0]
    cur = np.zeros(shape=(num_points, q_dim+1))
    cur[:,:q_dim] = path
    cur[1:,q_dim] = np.sum((cur[1:,:q_dim] - cur[:-1,:q_dim])**2, axis=1)**(1/2) # dists col
    nex = np.zeros(shape=(num_points, q_dim+1))

    # all the edges should be of length step_size=0.05
    assert(np.allclose(cur[1:,q_dim], np.asarray(step_size)))

    for i in range(num_iters):
        cumsums = cur[:num_points,q_dim].cumsum()
        path_len = cumsums[num_points-1]
        llen, rlen = params[i] * path_len
        llen, rlen = 0.01, 0.31
        # cumsums[lidx-1] <= llen < cumsums[lidx]
        # node_{lidx-1} <= left_endpoint < node_{lidx}
        lidx = np.searchsorted(cumsums[:num_points], llen, side='right')
        assert(lidx > 0)
        # cumsums[ridx-1] <= llen < cumsums[ridx]
        # node_{ridx-1} <= right_endpoint < node_{ridx}
        ridx = lidx + np.searchsorted(cumsums[lidx:num_points], rlen, side='right')
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

        print(f"\niter({i})")
        print(f"(lt: {lt}, rt: {rt})")
        print(f"(llen: {llen}, rlen: {rlen}, path_len: {path_len}")
        print(f"(lidx: {lidx}, ridx: {ridx})")
        for x in range(20):
            print(f"{x}: ", cur[x], cumsums[x])
        # exit(0)
        
        # way too close to endpoint nodes; float error my throw some shit
        if np.allclose(lq, cur[lidx-1,:q_dim]) or np.allclose(lq, cur[lidx,:q_dim]):
            print("too close left")
            assert False, "too close left"
            continue
        if np.allclose(rq, cur[ridx-1,:q_dim]) or np.allclose(rq, cur[ridx,:q_dim]):
            print("too close right")
            assert False, "too close right"
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
            print("collides")
            draw_line(get_high(lq), get_high(rq), 1, (0,0,1))
            draw_path(path)
            wait_for_user()
            exit(0)
            continue

        if (lidx == ridx-1):
            # TODO: resize array +1 when nodes are in adjacent edges
            assert False, "need to handle this stupid case"
            print("shit")
        else:
            assert(lidx < ridx-1)
            nex[:lidx] = cur[:lidx]

            nex[lidx,:q_dim] = lq
            nex[lidx,q_dim] = llen - cumsums[lidx-1]

            nex[lidx+1,:q_dim] = rq
            nex[lidx+1,q_dim] = rlen - cumsums[ridx-1]

            delta = lidx - ridx + 2
            nex_num_points = num_points + delta
            print(delta)
            nex[lidx+2:nex_num_points] = cur[ridx:num_points]
            nex[lidx+2, q_dim] = np.sum((nex[lidx+2,:q_dim] - nex[lidx+1,:q_dim])**2)**(1/2)

            print(ridx,num_points)
            print(lidx+2,nex_num_points)
            for x in range(20):
                print(f"{x}: ", nex[x])

            draw_line(get_high(lq), get_high(rq), 1, (0,1,0))
            draw_path(path)
            wait_for_user()
            exit(0)
            
            # cur_valid_idx = np.where(valid)[0]
            # print(cur_valid_idx)
            # print(cur_valid_idx[lidx])
            # cur[cur_valid_idx[lidx+1],:q_dim] = lq
            # cur[cur_valid_idx[lidx+2],:q_dim] = rq
            # valid[cur_valid_idx[lidx+3]:cur_valid_idx[ridx]] = False

            # cur_valid_idx = np.where(valid)[0]
            # print(cur_valid_idx)

            # BUG: I don't understand why this valid indexing is even correct (is it?)
            # Valid is still the original path array shape,
            # but lidx and ridx are indices defined for the valid portion of path.
            # Shouldn't it be valid[valid][lidx+2:ridx-1]?
            # However when I do this, no shortcuts are added and the smoothed
            # path is equal to the original path.
            # Regardless, I'm pretty sure this indexing is wrong.

            # original_valid_indices = np.where(valid)[0]
            # orig_lidx = original_valid_indices[lidx]
            # orig_ridx = original_valid_indices[ridx]

            # valid[orig_lidx+2:orig_ridx-1]=False

        # remove_all_debug()
        # draw_line(get_high(lq), get_high(rq), 1, (0,1,0))
        # draw_path(path)
        # wait_for_user()

        pass

    # print(cur)
    # print(valid)
    # exit(0)
    # print(params)
    # BUG: smoothing is clearly not working. It clips through walls and shit.
    smoothed_path = path[valid]

    exit(0)

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
