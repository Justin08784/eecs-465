import numpy as np

'''
Physical constants
'''
WALL_HEIGHT = 0.4
# NOTE:
# RRT has an easier time when MAX_LIN_ACCEL / MAX_LIN_VEL ≥ 2
# If max velocity is too high compared to lin accel, I think it has
# trouble avoiding "all actions result in unavoidable collision" states
# However, a higher ratio does seem to result in smoother and prettier paths
# (I recommend 2/2).
# A very low ratio, in contrast, would give the robot near-holonomic maneuverability.
MAX_LIN_ACCEL = 2
MAX_ANG_ACCEL = 2
MAX_LIN_VEL = 1
MAX_ANG_VEL = 0.7
# WARNING: hardcoded room dimensions; confirm in env_json of load_env
XLIMIT = 2.6
YLIMIT = 2.6

# Simulation
# NOTE: dt obviously affects performance a lot
# 0.02, 0.08 are both good values. 0.08 yields a tremendous
# speed increase over 0.02, while maintaining precision???
# dt = 0.02       # the resolution to which we are simulating
dt = 0.08

# NOTE: dt_sim exerts an unexpectedly large impact on runtime;
# when dt_sim increased from dt_sim = 0.2 to 1, we obtained a tremendous
# speed increase. This INCLUDES the 1/10 scale robot (which is the largest
# robot we have tested so far; it is the most computationally strenuous for our RRT).
# Other interesting consequences:
# 1. In extend_to(), simulate() became the dominant bottleneck (~90% runtime).
# At dt_sim = 0.2, collision_fn() was the bottleneck (~60% runtime).
# Speculation: an early collision_fn() == True stop saves much more work on a
# sim_state array with more NUM_SIM_STEPS??
dt_sim = 1    # minimum time interval to apply a control
NUM_SIM_STEPS = int(dt_sim/dt)
epsilon = 0.05  # if metric(s1, s2) < epsilon, then the states are "equivalent"

'''
Robot state
'''
# initial configuration
ROBOT_Z = WALL_HEIGHT/2
# s0 = np.array([2, -2, ROBOT_Z, np.pi/2], dtype=np.float64) # x, y, z, theta
s0 = np.array([2, -1.5, ROBOT_Z, np.pi/2], dtype=np.float64) # x, y, z, theta
# s0 = np.array([2, -2, ROBOT_Z, np.pi/2], dtype=np.float64) # x, y, z, theta
v0 = np.array([0, 0, 0, 0], dtype=np.float64) # v_x, v_y, v_z, \omega
u0 = np.array([0, 0, 0, 0], dtype=np.float64) # a_x, a_y, a_z, \alpha


# goal configuration
# sg = np.array([2, 2, ROBOT_Z, 0], dtype=np.float64) # x, y, theta
sg = np.array([1.5, 1, ROBOT_Z, 0], dtype=np.float64) # x, y, theta
# vg = np.array([0, 0, 0], dtype=np.float64) # v_x, v_y, \omega
CONTROL_LIN_ORI_RES = (45) * np.pi/180 # degrees (specify in parens)
CONTROL_LIN_MAG_RES = 1             # ms^-1
CONTROL_ANG_RES = 2
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
RAND_LEN = 4000

'''
Debugging
'''
PRINT_INTERVAL = 100
