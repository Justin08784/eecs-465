import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
import heapq
import itertools

#########################

import itertools
def main(screenshot=False):
    # grid details
    xlimit = 4.1 # x = [-xlimit, xlimit]
    ylimit = 2.1 # y = [-ylimit, ylimit]
    ang_res = np.pi/2
    lin_res = 0.1 # robot turns AWAY from goal direction momentarily, for some reason
    # BUG: (66, 14, 3) has a higher f(n) value than (66, 14, 1). Wtf?

    def to_idx(cx, cy, cr):
        return round((cx + xlimit)/lin_res),\
               round((cy + ylimit)/lin_res),\
               round(cr/ang_res)
    def to_coord(ix, iy, ir):
        return -xlimit + lin_res*ix,\
               -ylimit + lin_res*iy,\
               ang_res*ir


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
    xrange = np.arange(-xlimit, xlimit+lin_res, lin_res)
    yrange = np.arange(-ylimit, ylimit+lin_res, lin_res)
    rrange = np.arange(-np.pi, np.pi, ang_res)
    numx = len(xrange)
    numy = len(yrange)
    numr = len(rrange)

    # for (x,y) in itertools.product(xrange, yrange):
    #     draw_sphere_marker((x, y, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []

    start_config=np.array(start_config)
    goal_config=np.array(goal_config)
    start_config[2] %= 2*np.pi
    goal_config[2] %= 2*np.pi
    print("colllidin", collision_fn(start_config))

    xind, yind, rind = to_idx(start_config[0], start_config[1], start_config[2])
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
    four_connected = True
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
        assert False, "todo: eight_connected"
 
    n=0
    while frontier:
        exf, pos = heapq.heappop(frontier)
        exg = cost_so_far[pos]
        expandee[:] = pos

        nbrs[:,:] = expandee[:] + incrs
        nbrs[:,2] %= numr 
        in_bound =\
            (0<=nbrs[:,0])&(nbrs[:,0]<numx)&\
            (0<=nbrs[:,1])&(nbrs[:,1]<numy)
            # (0<=nbrs[:,4])&(nbrs[:,4]<numr)

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
        # print("goal\n", goal_config)
        print("expandee", exf, expandee, expandee_coords)
        heurs[:] = (
            (goal_config[0] - nbrs_coords[:, 0])**2+\
            (goal_config[1] - nbrs_coords[:, 1])**2+\
            np.minimum(
                abs(goal_config[2] - nbrs_coords[:, 2]),
                2*np.pi - abs(goal_config[2] - nbrs_coords[:, 2])
            )**2
        )**0.5
        # print("expandee_coords\n", expandee_coords)
        # print("nbrs\n", nbrs)
        # print("nbrs_coords\n", nbrs_coords)
        # print("costs\n", costs)
        # print("heurs\n", heurs)
        # print(in_bound)

        cands = np.arange(num_nbrs)
        for i in cands[in_bound]:
            nbr_pos = tuple(nbrs[i])
            nbr_cost = costs[i]
            nbr_heur = heurs[i]

            # print(nbrs_coords[i], "COLL", collision_fn(nbrs_coords[i]))
            if (collision_fn(nbrs_coords[i])):
                continue
            if (nbr_pos not in cost_so_far) or (nbr_cost < cost_so_far[nbr_pos]):
                cost_so_far[nbr_pos] = nbr_cost
                heapq.heappush(frontier, (nbr_heur + nbr_cost, nbr_pos))
                came_from[nbr_pos] = pos
        from pprint import pprint 
        # pprint(cost_so_far)
        # pprint(frontier)
        # pprint(came_from)
        if np.all(expandee == (66, 14, 0)):
            # print("ding on")
            # print("nbrs_coords\n", nbrs_coords)
            # print("nbrs\n", nbrs)
            # print("heurs\n", heurs)
            # print("costs\n", costs - exg)
            for x, st in frontier:
                # print(x, st, to_coord(st[0], st[1], st[2]), "prev:", came_from[st])
                pass
            # exit(0)

        if np.allclose(expandee_coords, goal_config, atol=1e-3):
            print("Goal reached!")
            break  # Stop the A* searchV
        print("digdfs")

    # print(start_config, goal_config)
    # print(expandee_coords)
    # exit(0)
    iidx = to_idx(start_config[0],start_config[1],start_config[2])
    fidx = to_idx(goal_config[0],goal_config[1],goal_config[2])


    path=[]
    cur=fidx
    while True:
        path.append(list(cur))
        nex = came_from[cur]
        if nex == iidx:
            path.append(list(nex))
            break
        cur = nex
    pprint(path[::-1])
    path=np.array(path, dtype=np.float64)
    path[:,0] = -xlimit + lin_res * path[:,0]
    path[:,1] = -ylimit + lin_res * path[:,1]
    path[:,2] = ang_res * path[:,2]
    path = path[::-1]


    print(path)
    # exit(0)



    # print("start", start_config)
    # print("idns", (xind, yind))
    # print("ind pos", to_coord(xind, yind))
    # print(start_config)

    # gp = lambda x : (-3.4+x,-1.4,0.0)
    # path = [gp(0.5*t) for t in range(10)]
    start_time = time.time()
    ### YOUR CODE HERE ###
    
    
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
