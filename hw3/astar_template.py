import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
import heapq
import itertools
from utils import draw_line
import itertools

#########################

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
    st_time = time.time()
    
    # grid details
    xlimit = 4.1 # x = [-xlimit, xlimit]
    ylimit = 2.1 # y = [-ylimit, ylimit]
    ang_res = np.pi/4
    lin_res = 0.1

    def to_idx(p):
        cx, cy, cr = p
        return round((cx + xlimit)/lin_res),\
               round((cy + ylimit)/lin_res),\
               round(cr/ang_res)
    def to_coord(p):
        ix, iy, ir = p
        return -xlimit + lin_res*ix,\
               -ylimit + lin_res*iy,\
               ang_res*ir

    xrange = np.arange(-xlimit, xlimit+lin_res, lin_res)
    yrange = np.arange(-ylimit, ylimit+lin_res, lin_res)
    rrange = np.arange(-np.pi, np.pi, ang_res)
    numx = len(xrange)
    numy = len(yrange)
    numr = len(rrange)

    start_config=np.array(start_config)
    goal_config=np.array(goal_config)
    start_config[2] %= 2*np.pi
    goal_config[2] %= 2*np.pi

    xind, yind, rind = to_idx(start_config)
    goal_closest = np.array(to_idx(goal_config))

    def cost(pos1, pos2):
        return (
            (pos2[0]-pos1[0])**2+\
            (pos2[1]-pos1[1])**2+\
            min(
                abs(pos2[2]-pos1[2]),
                2*np.pi - abs(pos2[2]-pos1[2])
            )**2
        )**0.5

    def heur(pos):
        return (
            (goal_config[0]-pos[0])**2+\
            (goal_config[1]-pos[1])**2+\
            min(
                abs(goal_config[2]-pos[2]),
                2*np.pi - abs(goal_config[2]-pos[2])
            )**2
        )**0.5

    np.set_printoptions(legacy='1.25')
    frontier = [[heur(start_config), (xind, yind, rind)]]
    cost_so_far = {(xind, yind, rind) : 0}
    came_from = {}

    expandee = np.zeros(3, dtype=np.int64)
    expandee_coords = np.zeros(3)
    four_connected = False
    collided = {}
    cnts = {}
    if four_connected:
        num_lin_nbrs, num_ang_nbrs = 4, 2
        num_nbrs = num_lin_nbrs+num_ang_nbrs
        nbrs = np.zeros((num_nbrs, 3), dtype=np.int64)
        nbrs_coords = np.zeros((num_nbrs, 3), dtype=np.float64)
        costs = np.zeros(num_nbrs, dtype=np.float64)
        heurs = np.zeros(num_nbrs, dtype=np.float64)
        in_bound = np.zeros(num_nbrs)
        pm = [-1,1]
        incrs = np.array([
            [1,0,0],
            [-1,0,0],
            [0,1,0],
            [0,-1,0],
            [0,0,1],
            [0,0,-1],
        ])

    else: # eight_connected
        num_lin_nbrs, num_ang_nbrs = 8, 2
        num_nbrs = num_lin_nbrs+num_ang_nbrs
        nbrs = np.zeros((num_nbrs, 3), dtype=np.int64)
        nbrs_coords = np.zeros((num_nbrs, 3), dtype=np.float64)
        costs = np.zeros(num_nbrs, dtype=np.float64)
        heurs = np.zeros(num_nbrs, dtype=np.float64)
        in_bound = np.zeros(num_nbrs)
        pm = [-1,1]
        incrs = np.array([
            [1,0,0],
            [-1,0,0],
            [0,1,0],
            [0,-1,0],
            [1,1,0],
            [1,-1,0],
            [-1,1,0],
            [-1,-1,0],
            [0,0,1],
            [0,0,-1],
        ])
        # assert False, "todo: eight_connected"

    n=0
    goal_reached = False
    # x,y positions of free/empty and obst/colliding nodes
    free_nodes = set()
    obst_nodes = set()

    # cache statistics
    num_expansions = 0
    num_cands = 0 # num nbrs considered in total
    cands_pruned = 0 # how good the heuristic is at pruning unpromising nbrs
    # TODO: 8-connected is better because the heuristic is able to
    # more effectively prune unpromising candidates (this is possible because
    # 8-connected allows more movement options to more closely approximate the
    # Euclidean metric best case)
    print("Connectivity:", 4 if four_connected else 8)
    while frontier:
        num_expansions += 1
        exf, pos = heapq.heappop(frontier)
        exg = cost_so_far[pos]
        expandee[:] = pos

        nbrs[:,:] = expandee[:] + incrs
        nbrs[:,2] %= numr
        in_bound =\
            (0<=nbrs[:,0])&(nbrs[:,0]<numx)&\
            (0<=nbrs[:,1])&(nbrs[:,1]<numy)

        expandee_coords[:] =\
            -xlimit + lin_res * expandee[0],\
            -ylimit + lin_res * expandee[1],\
            ang_res * expandee[2]
        nbrs_coords[:,0] = -xlimit + lin_res * nbrs[:,0]
        nbrs_coords[:,1] = -ylimit + lin_res * nbrs[:,1]
        nbrs_coords[:,2] = ang_res * nbrs[:,2]

        # g(n)
        costs[:] = exg + (
            (expandee_coords[0] - nbrs_coords[:, 0])**2+\
            (expandee_coords[1] - nbrs_coords[:, 1])**2+\
            np.minimum(
                abs(expandee_coords[2] - nbrs_coords[:, 2]),
                2*np.pi - abs(expandee_coords[2] - nbrs_coords[:, 2])
            )**2
        )**0.5

        # h(n)
        heurs[:] = (
            (goal_config[0] - nbrs_coords[:, 0])**2+\
            (goal_config[1] - nbrs_coords[:, 1])**2+\
            np.minimum(
                abs(goal_config[2] - nbrs_coords[:, 2]),
                2*np.pi - abs(goal_config[2] - nbrs_coords[:, 2])
            )**2
        )**0.5

        cands = np.arange(num_nbrs)
        w = 1.0 # 2 explores far fewer paths; but is optimality guarantee preserved?
        for i in cands[in_bound]:
            num_cands += 1
            nbr_pos = tuple(nbrs[i])
            nbr_cost = costs[i]
            nbr_heur = w * heurs[i]

            if (nbr_pos in cost_so_far) and (nbr_cost >= cost_so_far[nbr_pos]):
                cands_pruned += 1
                continue

            cache_ignore_rot = False
            if cache_ignore_rot:
                # aggressive optimization; don't key collide cache by rot; but might cause collision
                x,y,r = nbrs[i]
                k = (x,y)
                col=False
                if k in collided:
                    col=collided[k]
                    cnts[k]+=1
                else:
                    col=collision_fn(nbrs_coords[i])
                    collided[k]=col
                    cnts[k]=1
            else:
                # DO key collide cache by rot; but this is slower
                k = tuple(nbrs[i])
                if k in collided:
                    col=collided[k]
                    cnts[k]+=1
                else:
                    col=collision_fn(nbrs_coords[i])
                    collided[k]=col
                    cnts[k]=1

            if (col):
                obst_nodes.add(tuple(nbrs[i,:2]))
                continue

            cost_so_far[nbr_pos] = nbr_cost
            heapq.heappush(frontier, (nbr_heur + nbr_cost, nbr_pos))
            came_from[nbr_pos] = pos

        if all(expandee == goal_closest):
            print("Goal reached!")
            goal_reached = True
            break  # Stop the A* searchV

    print("Statistics:")
    print("  num_expansions", num_expansions)
    print(f"  num_cands: {num_cands} (pruned: {cands_pruned})")
    miss_cnt = len(cnts)
    total_cnt = sum(cnts.values())
    hit_cnt = total_cnt-miss_cnt
    print(f"  (collide cache) misses: {miss_cnt}, hits: {hit_cnt}, hit_rate: {hit_cnt/total_cnt}")

    if not goal_reached:
        print("No Solution Found.")
        exit(0)
    iidx = to_idx(start_config)
    fidx = to_idx(goal_config)
    print("  path_cost:", cost_so_far[fidx])


    path=[]
    cur=fidx
    while True:
        path.append(list(cur))
        nex = came_from[cur]
        if nex == iidx:
            path.append(list(nex))
            break
        cur = nex
    path=np.array(path, dtype=np.float64)
    path[:,0] = -xlimit + lin_res * path[:,0]
    path[:,1] = -ylimit + lin_res * path[:,1]
    path[:,2] = ang_res * path[:,2]
    path = path[::-1]
    print("  run_time:", time.time() - st_time)

    setz = lambda p, z : (p[0], p[1], z)
    def draw_path(path, col_code=(1,0,0), line_width=2, z=0.2):
        for i in range(len(path) - 1):
            p_i = path[i]
            p_f = path[i+1]
            draw_line(setz(p_i,z), setz(p_f,z), line_width, col_code)

    draw_path(path, col_code=(0,0,0), line_width=4, z=0.2)


    # visualize free and obstacle nodes
    for n in came_from:
        free_nodes.add((n[0], n[1]))

    # TODO: An (x,y) can be in BOTH colliding and non-colliding configuration depending on rotation angle
    # My guess: If an (x,y) has at least one colliding configuration, consider it colliding.
    # print("Intersection of free_nodes, obst_nodes:\n", free_nodes.intersection(obst_nodes))

    debug_nodes = False
    if debug_nodes:
        for n in free_nodes:
            if n in obst_nodes:
                continue
            p = to_coord((n[0], n[1], 0))
            draw_sphere_marker(setz(p, 0.1), 0.04, (0, 0, 1, 1))
        for n in obst_nodes:
            p = to_coord((n[0], n[1], 0))
            draw_sphere_marker(setz(p, 0.1), 0.04, (1, 0, 0, 1))
    
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
