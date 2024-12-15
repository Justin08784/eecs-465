import numpy as np
from utils import load_env, get_collision_fn_PR2, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, wait_if_gui, wait_for_user, joint_from_name, get_joint_info, get_link_pose, link_from_name
from pybullet_tools.utils import get_joints
from pybullet_tools.utils import set_joint_positions, \
    wait_if_gui, wait_for_duration, get_collision_fn, load_pybullet, get_pose, \
    get_bodies, get_body_name, set_pose, remove_all_debug
import myutils as my
import pybullet as p
import time
from pprint import pprint
from simulator import simulate
import itertools
import math
import constants as c
import random

# NOTE:
#  Create a helper that writes pre-set configs
# (e.g. drone dims; dt;...)

#NOTE:
# Good config for time performance:
# 1. dt = 0.08s
# 2. CONTROL_LIN_MAG_RES = 2
# 3.
'''
Initialization functions
'''
def create_drone(x, y, theta):
    # 1/9 is the largest that fits through channels, I think
    scale = 1/20

    half_extents = scale * np.array([4,3,1])
    # half_extents = scale * np.array([7.5,3,1])
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0, 1, 0, 1])
    # r=0.1
    # collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=0)
    # visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=^, rgbaColor=[1, 0, 0, 0.5])
    body_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=(x,y,c.ROBOT_Z),
                                baseOrientation=p.getQuaternionFromEuler((0,0,theta)))
    return body_id


def create_wall(x, y, theta, len):
    half_extents = [0.1, len / 2, c.WALL_HEIGHT / 2]
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 0.5])
    body_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=(x,y,c.WALL_HEIGHT/2),
                                baseOrientation=p.getQuaternionFromEuler((0,0,theta)))
    return body_id


def create_cylinder(x, y, r):
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=c.WALL_HEIGHT)
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=c.WALL_HEIGHT, rgbaColor=[1, 0, 0, 0.5])
    body_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=(x,y,c.WALL_HEIGHT/2),
                                baseOrientation=p.getQuaternionFromEuler((0,0,0)))
    return body_id

'''
Helpers
'''
def execute_trajectory(robot_id, states, dt=c.dt):
    for i in range(len(states)):
        x, y, z, theta = states[i,:4]
        set_pose(robot_id, ((x, y, z), p.getQuaternionFromEuler((0,0,theta))))
        time.sleep(dt)



'''
Simulated states
'''
sim_states = None

'''
Tree
'''
tree_len = None # num allocated entries (i.e. size of tree array)
tree_cur = None # num used entries; equiv, next free idx
state_tree = None

nbrs_of = None # maps each index to its nbrs

goal_reached = False
free = None

'''
Random tape
'''
rand_cur = None
state_rand = None

def fill_random(rand):
    # initialize poses
    rand[:,0] = np.random.uniform(low=-c.XLIMIT, high=c.XLIMIT, size=c.RAND_LEN)
    rand[:,1] = np.random.uniform(low=-c.YLIMIT, high=c.YLIMIT, size=c.RAND_LEN)
    rand[:,2] = c.ROBOT_Z
    rand[:,3] = np.random.uniform(low=0, high=2*np.pi, size=c.RAND_LEN)

    # initialize velocities
    v_mags = np.random.uniform(low=0, high=c.MAX_LIN_VEL, size=c.RAND_LEN)

    rand[:,4] = v_mags * np.cos(rand[:,3])
    rand[:,5] = v_mags * np.sin(rand[:,3])
    rand[:,6] = 0
    rand[:,7] = np.random.uniform(low=-c.MAX_ANG_VEL, high=c.MAX_ANG_VEL, size=c.RAND_LEN)

def init_globals():
    # For fixed rng (use this for perf testing)
    # np.random.seed(0)
    # random.seed(0)
    global sim_states
    sim_states = np.zeros((c.NUM_CONTROL_PRIMITIVES, c.NUM_SIM_STEPS, 8), dtype=np.float64)

    global tree_len
    tree_len = 128
    global tree_cur
    tree_cur = 1
    global state_tree
    state_tree = np.zeros((tree_len, 8), dtype=np.float64)
    state_tree[:,:3] = np.inf
    state_tree[0,c.IDX_POS] = c.s0
    state_tree[0,c.IDX_VEL] = c.v0
    
    global nbrs_of
    nbrs_of = {}
    nbrs_of[0] = []

    global rand_cur
    rand_cur = 0
    global state_rand
    state_rand = np.zeros((c.RAND_LEN, 8), dtype=np.float64)
    fill_random(state_rand)

def heur(states, dst, lin_w=1, ang_w=0.1):
    # NOTE: disabling ang error has better performance, for some reason
    # lin_w = 1
    # ang_w = 0
    # ang_w = 0.1
    return (
        # NOTE: if you change lin_error power from 2 to 12, it corrects
        # more aggressively at longer distances, leading to better perf
        lin_w*np.sum((states[:,:3] - dst[:3])**2, axis=1)+\
        ang_w*np.minimum(
            abs(states[:,3] - dst[3]),
            2*np.pi - abs(states[:,3] - dst[3])
        )**2
    )**0.5

# persistent locals for extend_to
tmp_curs = np.zeros(4, dtype=np.float64)
tmp_curv = np.zeros(4, dtype=np.float64)
DEBUG_PERF = False
def gettime():
    return time.time() if DEBUG_PERF else 0
def debug_time(sim, col, upd):
    total_time = sim + col + upd
    if upd == 0:
        print("upd_time zero: skipping")
        return
    scalar = 1/upd
    print("total_time = {%.6f} || sim : col : upd = {%.2f} : {%.2f} : {%.2f}" %\
        (total_time,
         sim * scalar,
         col * scalar,
         upd * scalar))
    
def extend_to(src_idx, dst, collision_fn, epsi, is_goal):
    '''
    src_idx: idx of source state in state_tree
    dst: destination state
    collision_fn: yep
    '''
    global state_tree, tree_cur, tree_len, free

    cur_root = src_idx # the initial (t = 0) idx of a trail
    tmp_curs[:] = state_tree[src_idx, c.IDX_POS]
    tmp_curv[:] = state_tree[src_idx, c.IDX_VEL]

    found = False
    MAX_NUM_EXTENDS = 200
    errors = np.zeros(c.NUM_CONTROL_PRIMITIVES, dtype=np.float64)
    errors[:] = np.inf
    prev_min_error = np.inf
    curr_min_error = np.inf
    best_min_error = np.inf
    argmin = np.zeros(c.NUM_CONTROL_PRIMITIVES, dtype=int)

    sim_time = 0
    col_time = 0
    upd_time = 0


    for i in range(MAX_NUM_EXTENDS):
        sim_start = gettime()
        simulate(c.CONTROL_SET, tmp_curs, tmp_curv, sim_states, c.NUM_SIM_STEPS)
        sim_time += gettime() - sim_start
        all_col = True

        col_start = gettime()
        # mincolt = []
        for ctrl in range(c.NUM_CONTROL_PRIMITIVES):
            col_t = c.NUM_SIM_STEPS # first colliding time step
            for t in range(c.NUM_SIM_STEPS):
                # look for first collision (if any)
                pos = sim_states[ctrl,t,:3]
                quat = p.getQuaternionFromEuler((0,0,sim_states[ctrl,t,3]))
                if collision_fn((pos, quat)):
                    # collided
                    col_t = t
                    break
            # mincolt.append(col_t)

            if col_t == 0:
                # first time step is already colliding. quit
                # we should exit at "Failed!"
                continue
            all_col = False

            # distance metric weights
            # NOTE: We only care about x,y position for goal. Getting a diversity of unique
            # orientations is only important for pathing to instrumental nodes, which may
            # require navigating narrow regions. If we insist on ang_w=0.1 for goal node,
            # the planner will get stuck trying to fix orientation (our goal region is quite
            # narrow) when it is not desired.
            dists_sq = heur(sim_states[ctrl,:col_t,:], dst, ang_w=0 if is_goal else 0.1)
            argmin[ctrl] = np.argmin(dists_sq)
            errors[ctrl] = dists_sq[argmin[ctrl]]
            if errors[ctrl] < c.epsilon:
                # print("Found!")
                found = True
                break
            if found:
                break
        # print(mincolt)
        col_time += gettime() - col_start
        # print(sim_states.shape, i)
        if all_col:
            # print("all col, breaking")
            # free[src_idx] = False
            if DEBUG_PERF:
                print(f"sim: {sim_time}, col: {col_time}, upd: {upd_time}")
                debug_time(sim_time, col_time, upd_time)
            return False

        upd_start = gettime()
        # pick control with minimum error
        opt_ctrl = np.argmin(errors)
        # time step in trail that had minimum error
        opt_idx = argmin[opt_ctrl]
        curr_min_error = errors[opt_ctrl]
        if curr_min_error >= 2 * best_min_error:
            # no improvement. quit
            # print("Failed!")
            if DEBUG_PERF:
                print(f"sim: {sim_time}, col: {col_time}, upd: {upd_time}")
                debug_time(sim_time, col_time, upd_time)
            return False
        prev_min_error = curr_min_error
        best_min_error = min(prev_min_error, best_min_error)

        # add optimal trails to state_tree
        trail_len = (opt_idx + 1)
        used_len = tree_cur + trail_len # tree_cur = current used length
        if used_len > tree_len:
            # NOTE: you should really put this tree manip logic in the same place.

            # exponentially resize tree
            expansion_factor = 2**math.ceil(math.log2((used_len) / tree_len))
            new_arr = np.zeros((state_tree.shape[0] * expansion_factor, state_tree.shape[1]))
            new_arr[:tree_cur, :] = state_tree[:tree_cur,:]
            new_arr[tree_cur:, :3] = np.inf
            state_tree = new_arr
            tree_len *= expansion_factor

            # new_free = ~np.zeros(tree_len, dtype=bool)
            # new_free[:tree_cur] = free[:tree_cur]
            # free = new_free
        state_tree[tree_cur:used_len] = sim_states[opt_ctrl, :trail_len, :]

        # add nbr relationships
        pre = cur_root
        for idx in range(tree_cur, used_len):
            nbrs_of[pre].append(idx)
            nbrs_of[idx] = [pre]
            pre = idx
        tree_cur = used_len

        cur_root = used_len - 1
        tmp_curs[:] = sim_states[opt_ctrl, opt_idx,:4]
        tmp_curv[:] = sim_states[opt_ctrl, opt_idx,4:]
        upd_time += gettime() - upd_start

        if found:
            if DEBUG_PERF:
                print(f"sim: {sim_time}, col: {col_time}, upd: {upd_time}")
                debug_time(sim_time, col_time, upd_time)
            return True
    return False
    print(time.time() - start)

# BUG:
# Sometimes the path goes OOB beyond the boundary walls. I think this happens
# because the boundary walls are so thin: if STEP_SIZE is large enough, then
# the wall can easily fit between two adjacent collision checks.

# BUG:
# Ok, I made the walls thicker. However, the bot keeps penetrating the North wall
# (the one next to the hole in the divider wall).

def main(env, screenshot=False, config=None):
    robot_id, collision_fn = env

    global goal_reached
    global rand_cur
    global state_tree, tree_len, tree_cur, free
    c.init_control_set()
    init_globals()

    choose_goal = False
    success = False
    i = 0                   # rand_idx, reset on refill
    overall_rand_idx = 0    # rand_idx, NOT reset on refill

    target = np.zeros(8, dtype=np.float64)
    hit = {}
    start = time.time()

    while not (choose_goal and success):
        if overall_rand_idx % c.PRINT_INTERVAL == 0:
            print("Target", overall_rand_idx, tree_cur)
        # draw_sphere_marker(dst[:3], 0.1, (0, 1, 0, 1))
        choose_goal = random.random() < c.GOAL_BIAS

        if choose_goal:
            target[:4] = c.sg
        else:
            target[:4] = state_rand[i,:4]
            i+=1
            overall_rand_idx+=1
        dists_sq = heur(state_tree, target)
        # dirs = target[:2] - state_tree[:, :2]
        # dists = np.linalg.norm(dirs, axis=1)
        # dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # vels = state_tree[:, 4:6]
        # ja = np.sum(vels * dirs, axis=1) / np.sum(dirs * dirs, axis=1)

        # stopping_distances = 1.5 * (ja**2)/c.MAX_LIN_ACCEL

        # valid = (stopping_distances <= dists) | (ja < 0)

        if i >= c.RAND_LEN:
            # refill ranndom tape
            fill_random(state_rand)
            i = 0
            print("refill")

        # NOTE: for cur_near, we first identify the closest state to the target (i.e. lowest metric value).
        # Then, we randomly select from all states whose metric value is at most 0.35 worse than that of
        # the closest state. This helps tremendously to path through narrow passages, where RRT can get stuck
        # if it repeatedly chooses the closest cur_near in the passage. By picking a bit behind, we introduce
        # diversity and increase possibility of breakthrough (i.e. combat planner stagnation).
        # Note, this does make path noticably more chaotic, which may be undesirable for smaller robots like
        # 1/20 scale, for which narrow passage pathing isn't too problematic.

        # reall_valid = valid & (dists_sq <= 0.175 + np.min(dists_sq))
        # if not np.any(reall_valid):
        #     continue
        # Approach 1: randomly choose among adequate nodes
        # cur_near=np.random.choice(np.arange(dists_sq.shape[0])\
        #                           [reall_valid])
        cur_near=np.random.choice(np.arange(dists_sq.shape[0])\
                                  [dists_sq <= 0.175 + np.min(dists_sq)])
        # Approach 2: pick node with probability proportional to one-sided normal pdf taking min_dist as the mean
        # min_dist = np.min(dists_sq)
        # adequate = np.arange(dists_sq.shape[0])[dists_sq <= 0.35 + min_dist]
        # weights = np.exp(-0.5*(dists_sq[adequate] - min_dist))
        # weights /= np.sum(weights)
        # cur_near=np.random.choice(adequate, p=weights)
        # Approach 3: greedy
        #cur_near = np.argmin(dists_sq)

        if cur_near not in hit:
            hit[int(cur_near)] = 1
        else:
            hit[int(cur_near)] += 1

        # success = extend_to(cur_near, target, collision_fn, c.epsilon, debug=i-1==36)
        success = extend_to(cur_near, target, collision_fn, c.epsilon, choose_goal)
        # success = extend_to(cur_near, target, collision_fn, c.epsilon, debug=i==100)
    print("rrt runtime:", time.time() - start)
    print(state_tree[:,4:6])
    # pprint(hit)

    cur = tree_cur - 1
    path=[]
    while True:
        path.append(state_tree[cur,:4])
        # parent of cur; we can do this thanks to topological ordering
        cur = nbrs_of[cur][0]
        if cur == 0:
            break
    path = np.array(path[::-1])

    def draw_path(path, col_code=(0,1,0), line_width=3):
        for i in range(len(path) - 1):
            line_start = path[i]
            line_end = path[i+1]
            line_color = col_code # R, G, B
            draw_line(line_start, line_end, line_width, line_color)

    set_pose(robot_id, (c.s0[:3], p.getQuaternionFromEuler((0,0,c.s0[3]))))

    # draw tree edges. WARNING: Destructive: destroys nbrs_of
    # for lidx in range(tree_len):
    #     if lidx not in nbrs_of:
    #         continue
    #     for ridx in nbrs_of[lidx]:
    #         line_start = state_tree[lidx,:3]
    #         line_end = state_tree[ridx,:3]

    #         line_width = 1
    #         line_color = (1, 0, 0) # R, G, B

    #         if ridx not in nbrs_of:
    #             continue
    #         nbrs_of[ridx].remove(lidx)
    #         if not nbrs_of[ridx]:
    #             nbrs_of.pop(ridx)

    #         draw_line(line_start, line_end, line_width, line_color)

    draw_path(path[:,:3])
    # while True:
    #     wait_for_user()
    # execute_trajectory(robot_id, path)
    remove_all_debug()


if __name__ == '__main__':
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and floor/walls 
    _, obstacles = load_env('envs/2D_drone.json')
    robot_id = create_drone(c.s0[0], c.s0[1], c.s0[3])
    obstacle_ids = list(obstacles.values())
    assert(not get_joints(robot_id))

    # add shape obstacles
    obstacle_ids.append(create_wall(-2.2,0,np.pi/2,0.6))
    obstacle_ids.append(create_wall(0.75,0,np.pi/2,3.5))
    obstacle_ids.append(create_wall(-2.5,0,0,5))
    obstacle_ids.append(create_wall(1.4,1.75,0,1.5))
    obstacle_ids.append(create_cylinder(0, -1.25, 0.5))
    obstacle_ids.append(create_cylinder(0, 1.25, 0.5))

    collision_fn = my.get_collision_fn(
        robot_id,
        obstacle_ids
    )

    num_runs = 10
    for i in range(num_runs):
        main(env=(robot_id, collision_fn))

    # Keep graphics window opened
    wait_if_gui()
    disconnect()
