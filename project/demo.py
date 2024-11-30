import numpy as np
from utils import load_env, get_collision_fn_PR2, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, wait_if_gui, wait_for_user, joint_from_name, get_joint_info, get_link_pose, link_from_name
from pybullet_tools.utils import get_joints
from pybullet_tools.utils import set_joint_positions, \
    wait_if_gui, wait_for_duration, get_collision_fn, load_pybullet, get_pose, \
    get_bodies, get_body_name, set_pose
import myutils as my
import pybullet as p
import time
from pprint import pprint
from simulator import simulate

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
MAX_ANG_ACCEL = 0.2

# WARNING: hardcoded room dimensions; confirm in env_json of load_env
XLIMIT = 2.6
YLIMIT = 2.6
dt = 0.01       # the resolution to which we are simulating
dt_ctrl = 0.1   # minimum time interval to apply a control

'''
Robot state
'''
s0 = np.array([2, -2, 0], dtype=np.float64) # x, y, theta
v0 = np.array([0, 0, 0], dtype=np.float64) # v_x, v_y, \omega
u0 = np.array([0, 0.1, 0], dtype=np.float64) # a_x, a_y, \alpha


'''
Tree
'''
TREE_LEN = 16
state_tree = np.zeros((TREE_LEN, 6), dtype=np.float64)
state_tree[:] = np.inf
nbrs_of = {} # maps each index to its nbrs
GOAL_BIAS = 0.1
STEP_SIZE = 0.1


def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and floor/walls 
    _, obstacles = load_env('envs/2D_drone.json')
    robot_id = create_drone(2, -2, 0)
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
            x, y, theta = states[i]
            set_pose(robot_id, ((x, y, WALL_HEIGHT/2), p.getQuaternionFromEuler((0,0,theta))))
            time.sleep(dt)

    # print(">>>>")
    # print(collision_fn(((-2,0.29,0.2), (0,0,0,1.0))))
    # print("<<<<")

    global s0
    global v0
    global u0
    global dt
    num_states = 1000
    states = np.zeros((num_states, 3), dtype=np.float64)
    simulate(states, s0, v0, u0, num_states, dt)

    wait_for_user()
    execute_trajectory(states, dt)

    exit(0)

    # define active DoFs
    joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit, get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}

    # set up collision function
    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    print("You can ignore the b3Printf warning messages")
    



    """ Example: check if a single joint position is within its joint limit
        In this example, we examine if a joint value for 'l_shoulder_pan_joint' is within its joint limit.
    """
    print("Example joint limit checking:")
    wait_for_user()
    # get corresponding joint lower and upper limits
    lspj_low_lim, lspj_up_lim = joint_limits['l_shoulder_pan_joint']
    # toy joint position
    lspj_cfg = 1.567
    print("Is joint within limit? ", "yes" if (lspj_cfg < lspj_up_lim and lspj_cfg > lspj_low_lim) else "no")
    print("=======================================")


    """ Example: convert between numpy arrays and tuples/lists
        In this example, we show how to convert a list to a numpy array, and how to convert a numpy array to tuple/list
    """
    robot_config = [-0.160, 0.075, -1.008, 0.000, 0.000, -0.110, 0.000]
    robot_config_arr = np.array(robot_config)
    robot_config_tuple = tuple(robot_config_arr)
    robot_config_list = list(robot_config_arr)


    """ Example: check robot collision
        In this example, we show how to check if a given robot configuration is in collision with itself and the world
    """
    print("Example robot collision checking: ")
    wait_for_user()
    # toy configuration
    robot_config_collide = (0.98, 1.190, -1.548, 1.557, -1.320, -0.193)
    # collision checker
    print("Robot in collision? ", "yes" if collision_fn(robot_config_collide) else "no")
    print("=======================================")
    

    """ Example: construct a path and execute it in the visualizer
        In this example, we show how to construct a path that can be visualized in the visualizer
    """
    print("Example path construction and execution")
    wait_for_user()
    # initialize path list
    path = []
    # append waypoints to path
    path.append([0.5218229734182527, 1.1693158423035832, -1.5186036819787623, 1.587179348050579, -1.277932523835633, -0.24223835168059277])
    path.append([0.5481083947124124, 1.1444021062720668, -1.483196288637991, 1.6235298884965914, -1.2272629238425585, -0.30178605295425226])
    path.append([0.5612511053594922, 1.1319452382563084, -1.4654925919676054, 1.6417051587195977, -1.2019281238460213, -0.331559903591082])
    path.append([0.5875365266536519, 1.1070315022247916, -1.4300851986268341, 1.6780556991656101, -1.151258523852947, -0.3911076048647415])
    path.append([0.6006792373007317, 1.0945746342090332, -1.4123815019564485, 1.6962309693886164, -1.1259237238564097, -0.42088145550157124])
    path.append([0.6138219479478115, 1.0821177661932748, -1.3946778052860629, 1.7144062396116226, -1.1005889238598725, -0.450655306138401])
    path.append([0.6269646585948914, 1.0696608981775164, -1.3769741086156773, 1.7325815098346289, -1.0752541238633353, -0.48042915677523074])
    path.append([0.6401073692419712, 1.057204030161758, -1.3592704119452916, 1.7507567800576351, -1.049919323866798, -0.5102030074120605])
    path.append([0.6663927905361309, 1.0322902941302412, -1.3238630186045204, 1.7871073205036476, -0.9992497238737237, -0.5697507086857199])
    path.append([0.6926782118302905, 1.0073765580987244, -1.2884556252637491, 1.8234578609496601, -0.9485801238806493, -0.6292984099593792])
    path.append([0.7058209224773704, 0.9949196900829661, -1.2707519285933635, 1.8416331311726664, -0.9232453238841121, -0.6590722605962089])
    # execute Path
    execute_trajectory(PR2, joint_idx, path, sleep=0.1)
    print("=======================================")


    """ Example: Draw a sphere
        In this example, we show how to draw a sphere with specified position and appearance
    """
    print("Example: draw a sphere")
    wait_for_user()
    sphere_position = (0, 0, 1)
    sphere_radius = 0.1
    sphere_color = (1, 0, 1, 0.5) # R, G, B, A
    draw_sphere_marker(sphere_position, sphere_radius, sphere_color)
    print("=======================================")


    """ Example: Draw a line
        In this example, we show how to draw a sphere with specified position and appearance
    """
    print("Example: draw a line")
    wait_for_user()
    line_start = (0.5, 0.5, 0.5)
    line_end = (0.7, 0.7, 0.7)
    line_width = 10
    line_color = (1, 0, 0) # R, G, B
    draw_line(line_start, line_end, line_width, line_color)
    print("=======================================")


    """ Example: Get the position of the tip of the PR2's left gripper
    """

    ee_pose = get_link_pose(PR2, link_from_name(PR2, 'l_gripper_tool_frame'))
    #ee_pose[0] is the translation of the left gripper tool frame
    #ee_pose[1] is the rotation (represented as a quaternion the left gripper tool frame), we don't need this
    print("Example: Get the position of the PR2's left gripper")
    wait_for_user()
    print(ee_pose[0])
    

    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
