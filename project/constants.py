import numpy as np

'''
Physical constants
'''
WALL_HEIGHT = 0.4
MAX_LIN_ACCEL = 1
MAX_ANG_ACCEL = 1
MAX_LIN_VEL = 2 # does nothing currently
MAX_ANG_VEL = 2 # does nothing currently
# WARNING: hardcoded room dimensions; confirm in env_json of load_env
XLIMIT = 2.6
YLIMIT = 2.6

# Simulation
dt = 0.01       # the resolution to which we are simulating
dt_sim = 0.1    # minimum time interval to apply a control
NUM_SIM_STEPS = int(dt_sim/dt)
epsilon = 0.01  # if metric(s1, s2) < epsilon, then the states are "equivalent"

'''
Robot state
'''
# initial configuration
ROBOT_Z = WALL_HEIGHT/2
s0 = np.array([2, -2, ROBOT_Z, 0], dtype=np.float64) # x, y, z, theta
v0 = np.array([-0.2, 0.2, 0, 0], dtype=np.float64) # v_x, v_y, v_z, \omega
u0 = np.array([0, 0, 0, 0], dtype=np.float64) # a_x, a_y, a_z, \alpha


# goal configuration
sg = np.array([2, 2, ROBOT_Z, 0], dtype=np.float64) # x, y, theta
# vg = np.array([0, 0, 0], dtype=np.float64) # v_x, v_y, \omega
CONTROL_LIN_ORI_RES = (45) * np.pi/180 # degrees (specify in parens)
CONTROL_LIN_MAG_RES = 0.5              # ms^-1
CONTROL_ANG_RES = 1
CONTROL_SET = None
NUM_CONTROL_PRIMITIVES = None

def init_control_set():
    global CONTROL_SET
    global NUM_CONTROL_PRIMITIVES
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
    NUM_CONTROL_PRIMITIVES = CONTROL_SET.shape[0]
    # print(CONTROL_SET.shape)
    # print(CONTROL_SET)

    # import matplotlib.pyplot as plt
    # plt.scatter(CONTROL_SET[:num_lins,0], CONTROL_SET[:num_lins,1])
    # plt.show()

'''
Tree
'''
GOAL_BIAS = 0.1
STEP_SIZE = 0.1

IDX_POS = np.arange(4)
IDX_VEL = np.arange(4, 8)


'''
Random tape
'''
RAND_LEN = 1000
