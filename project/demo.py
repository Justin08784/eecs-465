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

'''
Initialization functions
'''
def create_drone(x, y, theta):
    scale = 1/20
    half_extents = scale * np.array([4,3,1])
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0, 1, 0, 1])
    body_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=(x,y,WALL_HEIGHT/2),
                                baseOrientation=p.getQuaternionFromEuler((0,0,theta)))
    return body_id


def create_wall(x, y, theta, len):
    half_extents = [0.1, len / 2, WALL_HEIGHT / 2]
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 0.5])
    body_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=(x,y,WALL_HEIGHT/2),
                                baseOrientation=p.getQuaternionFromEuler((0,0,theta)))
    return body_id


def create_cylinder(x, y, r):
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=WALL_HEIGHT)
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=WALL_HEIGHT, rgbaColor=[1, 0, 0, 0.5])
    body_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=(x,y,WALL_HEIGHT/2),
                                baseOrientation=p.getQuaternionFromEuler((0,0,0)))
    return body_id


'''
Physical constants
'''
WALL_HEIGHT = 0.4
MAX_LIN_ACCEL = 1
MAX_ANG_ACCEL = 1
MAX_LIN_VEL = 2
MAX_ANG_VEL = 2

# WARNING: hardcoded room dimensions; confirm in env_json of load_env
XLIMIT = 2.6
YLIMIT = 2.6
dt = 0.01       # the resolution to which we are simulating
dt_sim = 0.1    # minimum time interval to apply a control
epsilon = 0.01  # if metric(s1, s2) < epsilon, then the states are "equivalent"

'''
Robot state
'''
# initial configuration
ROBOT_Z = WALL_HEIGHT/2
s0 = np.array([2, -2, ROBOT_Z, 0], dtype=np.float64) # x, y, z, theta
v0 = np.array([0, 0, 0, 0], dtype=np.float64) # v_x, v_y, v_z, \omega
u0 = np.array([0, 0, 0, 0], dtype=np.float64) # a_x, a_y, a_z, \alpha


# goal configuration
sg = np.array([2, 2, ROBOT_Z, 0], dtype=np.float64) # x, y, theta
# vg = np.array([0, 0, 0], dtype=np.float64) # v_x, v_y, \omega
CONTROL_LIN_ORI_RES = (45) * np.pi/180 # degrees (specify in parens)
CONTROL_LIN_MAG_RES = 0.5              # ms^-1
CONTROL_ANG_RES = 1
CONTROL_SET = None

def init_control_set():
    global CONTROL_SET
    # Part 1: initialize linear controls
    num_lin_oris = int(2*np.pi / CONTROL_LIN_ORI_RES)
    num_lin_mags = int(MAX_LIN_ACCEL / CONTROL_LIN_MAG_RES)

    # compute non-zero linear control magnitudes
    mags_base = np.arange(1, num_lin_mags + 1)
    lin_mags = np.concatenate((mags_base, -mags_base)) * CONTROL_LIN_MAG_RES

    # compute linear controls with non-zero magnitudes
    lin_oris = CONTROL_LIN_ORI_RES * np.arange(num_lin_oris)
    xs = np.outer(lin_mags, np.cos(lin_oris)).flatten()
    ys = np.outer(lin_mags, np.sin(lin_oris)).flatten()

    # add the null linear control (i.e. linear control with zero magnitude)
    xs = np.insert(xs, 0, 0)
    ys = np.insert(ys, 0, 0)
    lin_vs = np.column_stack([xs, ys])

    # Part 2: initialize angular controls
    num_angv = int(MAX_ANG_ACCEL / CONTROL_ANG_RES)
    ang_vs = CONTROL_ANG_RES * np.arange(-num_angv, num_angv + 1)
    # print(ang_vs, ang_vs.shape)
    # print(lin_mags, lin_mags.shape)

    num_lins = lin_vs.shape[0]
    num_angs = ang_vs.shape[0]
    CONTROL_SET = np.zeros((num_angs, num_lins, 3))
    CONTROL_SET[:, :, :2] = lin_vs
    CONTROL_SET[:, :, 2] = ang_vs[:, np.newaxis]
    CONTROL_SET = CONTROL_SET.reshape(-1, CONTROL_SET.shape[2])
    # print(CONTROL_SET.shape)
    # print(CONTROL_SET)

    # import matplotlib.pyplot as plt
    # plt.scatter(CONTROL_SET[:num_lins,0], CONTROL_SET[:num_lins,1])
    # plt.show()

init_control_set()

'''
Simulated states
'''
NUM_SIM_STEPS = int(dt_sim/dt)
# NUM_SIM_STEPS = 300
sim_states = np.zeros((CONTROL_SET.shape[0], NUM_SIM_STEPS, 8), dtype=np.float64)


'''
Tree
'''
GOAL_BIAS = 0.1
STEP_SIZE = 0.1
tree_len = 16 # num allocated entries (i.e. size of tree array)
tree_cur = 0  # num used entries; equiv, next free idx

state_tree = np.zeros((tree_len, 8), dtype=np.float64)
IDX_POS = np.arange(4)
IDX_VEL = np.arange(4, 8)
state_tree[:] = np.inf
state_tree[0,IDX_POS] = s0
state_tree[0,IDX_VEL] = v0
tree_cur = 1

nbrs_of = {} # maps each index to its nbrs
nbrs_of[0] = []

goal_reached = False

'''
Random tape
'''
RAND_LEN = 1000
rand_cur = 0
state_rand = np.zeros((RAND_LEN, 8), dtype=np.float64)

def fill_random(rand):
    # initialize poses
    rand[:,0] = np.random.uniform(low=-XLIMIT, high=XLIMIT, size=RAND_LEN)
    rand[:,1] = np.random.uniform(low=-YLIMIT, high=YLIMIT, size=RAND_LEN)
    rand[:,2] = ROBOT_Z
    rand[:,3] = np.random.uniform(low=0, high=2*np.pi, size=RAND_LEN)

    # initialize velocities
    v_mags = np.random.uniform(low=0, high=MAX_LIN_VEL, size=RAND_LEN)

    rand[:,4] = v_mags * np.cos(rand[:,3])
    rand[:,5] = v_mags * np.sin(rand[:,3])
    rand[:,6] = 0
    rand[:,7] = np.random.uniform(low=-MAX_ANG_VEL, high=MAX_ANG_VEL, size=RAND_LEN)
fill_random(state_rand)


# BUG:
# Sometimes the path goes OOB beyond the boundary walls. I think this happens
# because the boundary walls are so thin: if STEP_SIZE is large enough, then
# the wall can easily fit between two adjacent collision checks.

def main(screenshot=False):
    global dt
    global dt_sim
    global s0
    global v0
    global u0

    # initialize PyBullet
    connect(use_gui=True)
    # load robot and floor/walls 
    _, obstacles = load_env('envs/2D_drone.json')
    robot_id = create_drone(s0[0], s0[1], s0[2])
    obstacle_ids = list(obstacles.values())
    assert(not get_joints(robot_id))

    # add shape obstacles
    obstacle_ids.append(create_wall(-2.2,0,np.pi/2,0.6))
    obstacle_ids.append(create_wall(0.75,0,np.pi/2,3.5))
    obstacle_ids.append(create_cylinder(0, -1.25, 0.5))

    collision_fn = my.get_collision_fn(
        robot_id,
        obstacle_ids
    )

    '''
    Helpers
    '''
    def execute_trajectory(states, dt):
        for i in range(len(states)):
            x, y, z, theta = states[i,:4]
            set_pose(robot_id, ((x, y, z), p.getQuaternionFromEuler((0,0,theta))))
            time.sleep(dt)


    # num_states = 1000
    tmp_sg = np.array([1, -1, ROBOT_Z, 0], dtype=np.float64) # x, y, theta
    draw_sphere_marker(tmp_sg[:3], 0.1, (0, 1, 0, 1))
    # free = ~np.zeros(CONTROL_SET.shape[0])
    curs = s0.copy()
    curv = v0.copy()
    found = False
    start = time.time()
    MAX_NUM_EXTENDS = 200
    min_cidx = -1
    errors = np.zeros(CONTROL_SET.shape[0], dtype=np.float64)
    errors[:] = np.inf
    argmin = np.zeros(CONTROL_SET.shape[0], dtype=int)

    slst = [curs]
    vlst = [curv]
    for i in range(MAX_NUM_EXTENDS):
        simulate(CONTROL_SET, curs, curv, sim_states, NUM_SIM_STEPS, dt)
        for c in range(CONTROL_SET.shape[0]):
            col_t = NUM_SIM_STEPS # first colliding time step
            for t in range(NUM_SIM_STEPS):
                # look for first collision (if any)
                pos = sim_states[c,t,:3]
                quat = p.getQuaternionFromEuler((0,0,sim_states[c,t,3]))
                if collision_fn((pos, quat)):
                    # collided
                    col_t = t
                    break

            dists_sq = np.sum((sim_states[c,:col_t,:3] - tmp_sg[:3])**2, axis=1)**(1/2)
            argmin[c] = np.argmin(dists_sq)
            errors[c] = dists_sq[argmin[c]]
            # print("\nIteration")
            # print(sim_states[c,:col_t,:3])
            # print(dists_sq)
            if errors[c] < epsilon:
                print("Found!")
                min_cidx = c
                found = True
                break
            if found:
                break

        opt_ctrl = np.argmin(errors)
        curs[:] = sim_states[opt_ctrl, argmin[opt_ctrl],:4]
        curv[:] = sim_states[opt_ctrl, argmin[opt_ctrl],4:]

        slst.append(list(curs))
        vlst.append(list(curv))
        if found:
            break
    print(time.time() - start)
    from pprint import pprint
    pprint(slst)


    # for c in range(CONTROL_SET.shape[0]):
    # for c in min_cidxs:
    #     print(f"Executing control {c}: {CONTROL_SET[c]}")
    execute_trajectory(np.array(slst), dt_sim)
    exit(0)

    # print(">>>>")
    # print(collision_fn(((-2,0.29,0.2), (0,0,0,1.0))))
    # print("<<<<")
    global goal_reached
    global rand_cur
    global state_tree, tree_len, tree_cur

    start = time.time()
    while (not goal_reached):
        if rand_cur >= RAND_LEN:
            # refill rand array
            fill_random(state_rand)
            rand_cur = 0

        import random
        if random.random() < GOAL_BIAS:
            # use goal target
            cur_tgt = sg[:3]
        else:
            # use random target
            cur_tgt = state_rand[rand_cur,:3]
            rand_cur += 1

        dists_sq = np.sum((state_tree[:tree_cur,:3] - cur_tgt)**2, axis=1)**(1/2)
        min_idx = int(np.argmin(dists_sq))
        cur_near = state_tree[min_idx,:3]
        uvec = (cur_tgt - cur_near)/dists_sq[min_idx]

        # print(cur_tgt, cur_near, uvec)
        # print("dists", dists_sq[0])
        # draw_sphere_marker(get_high(cur_near), 0.1, (0, 1, 0, 1))
        # draw_sphere_marker(get_high(cur_tgt), 0.1, (0, 1, 0, 1))
        max_step = int(dists_sq[min_idx]/STEP_SIZE)
        prev_idx = min_idx
        cur_idx = tree_cur
        for t in range(1, max_step + 1):
            pt = cur_near+t*STEP_SIZE*uvec
            if (collision_fn(
                (
                    pt,
                    (0,0,0,1.0)
                ))):
                break
            goal_reached = np.sum((pt - sg[:3])**2)**(0.5) < STEP_SIZE

            if (tree_cur >= tree_len):
                # resize tree array
                new_arr = np.zeros((state_tree.shape[0] * 2, state_tree.shape[1]))
                new_arr[:tree_cur, :] = state_tree
                state_tree = new_arr
                tree_len *= 2

            tree_cur += 1

            state_tree[cur_idx,:3] = pt
            # init[cur_idx] = True
            nbrs_of[prev_idx].append(cur_idx)
            nbrs_of[cur_idx] = [prev_idx]

            prev_idx = cur_idx
            cur_idx += 1
            # draw_sphere_marker(get_high(pt), 0.1, (1, 0, 0, 1))
    print("runtime:", time.time() - start, tree_len)

    # construct path
    cur = tree_cur - 1
    path=[]
    while True:
        path.append(state_tree[cur,:3])
        # parent of cur; we can do this thanks to topological ordering
        cur = nbrs_of[cur][0]
        if cur == 0:
            break
    path = path[::-1]
    # from pprint import pprint
    # pprint(path[::-1])

    # draw tree edges. WARNING: Destructive: destroys nbrs_of
    for lidx in range(tree_len):
        if lidx not in nbrs_of:
            continue
        for ridx in nbrs_of[lidx]:
            line_start = state_tree[lidx,:3]
            line_end = state_tree[ridx,:3]

            line_width = 1
            line_color = (1, 0, 0) # R, G, B

            if ridx not in nbrs_of:
                continue
            nbrs_of[ridx].remove(lidx)
            if not nbrs_of[ridx]:
                nbrs_of.pop(ridx)

            draw_line(line_start, line_end, line_width, line_color)

    wait_for_user()

    def draw_path(path, col_code=(0,1,0), line_width=3):
        for i in range(len(path) - 1):
            line_start = path[i]
            line_end = path[i+1]
            line_color = col_code # R, G, B
            draw_line(line_start, line_end, line_width, line_color)

    path = np.array(path)
    draw_path(path)
    wait_for_user()

    exit(0)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
