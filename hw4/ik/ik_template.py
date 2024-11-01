import numpy as np
from pybullet_tools.utils import connect, disconnect, set_joint_positions, wait_if_gui, set_point, load_model,\
                                 joint_from_name, link_from_name, get_joint_info, HideOutput, get_com_pose, wait_for_duration
from pybullet_tools.transformations import quaternion_matrix
from pybullet_tools.pr2_utils import DRAKE_PR2_URDF
import time
import sys
### YOUR IMPORTS HERE ###

#########################

from utils import draw_sphere_marker

def get_ee_transform(robot, joint_indices, joint_vals=None):
    # returns end-effector transform in the world frame with input joint configuration or with current configuration if not specified
    if joint_vals is not None:
        set_joint_positions(robot, joint_indices, joint_vals)
    ee_link = 'l_gripper_tool_frame'
    pos, orn = get_com_pose(robot, link_from_name(robot, ee_link))
    res = quaternion_matrix(orn)
    res[:3, 3] = pos
    return res

def get_joint_axis(robot, joint_idx):
    # returns joint axis in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    R_W_J = H_W_J[:3, :3]
    joint_axis_local = np.array(j_info.jointAxis)
    joint_axis_world = np.dot(R_W_J, joint_axis_local)
    return joint_axis_world

def get_joint_position(robot, joint_idx):
    # returns joint position in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    j_world_posi = H_W_J[:3, 3]
    return j_world_posi

def set_joint_positions_np(robot, joints, q_arr):
    # set active DOF values from a numpy array
    q = [q_arr[0, i] for i in range(q_arr.shape[1])]
    set_joint_positions(robot, joints, q)


def get_translation_jacobian(robot, joint_indices):
    J = np.zeros((3, len(joint_indices)))
    ### YOUR CODE HERE ###
    ee_pos = get_ee_transform(robot, joint_indices, joint_vals=None)[:3,3]
    joint_axes = np.array([get_joint_axis(robot, i) for i in joint_indices])
    joint_posi = np.array([get_joint_position(robot, i) for i in joint_indices])
    J = np.cross(joint_axes,  ee_pos - joint_posi, axis=1).T



    ### YOUR CODE HERE ###
    return J

def get_jacobian_pinv(J):
    J_pinv = []
    ### YOUR CODE HERE ###
    # TODO: Why can't we just use pinv?
    # J_pinv1 = np.linalg.pinv(J)
    lam = 1e-12
    J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.identity(J.shape[0]))



    ### YOUR CODE HERE ###
    return J_pinv

def tuck_arm(robot):
    joint_names = ['torso_lift_joint','l_shoulder_lift_joint','l_elbow_flex_joint',\
        'l_wrist_flex_joint','r_shoulder_lift_joint','r_elbow_flex_joint','r_wrist_flex_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    set_joint_positions(robot, joint_idx, (0.24,1.29023451,-2.32099996,-0.69800004,1.27843491,-2.32100002,-0.69799996))

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Specify which target to run:")
        print("  'python3 ik_template.py [target index]' will run the simulation for a specific target index (0-4)")
        exit()
    test_idx = 0
    try:
        test_idx = int(args[0])
    except:
        print("ERROR: Test index has not been specified")
        exit()

    # initialize PyBullet
    connect(use_gui=True, shadows=False)
    # load robot
    with HideOutput():
        robot = load_model(DRAKE_PR2_URDF, fixed_base=True)
        set_point(robot, (-0.75, -0.07551, 0.02))
    tuck_arm(robot)
    # define active DoFs
    joint_names =['l_shoulder_pan_joint','l_shoulder_lift_joint','l_upper_arm_roll_joint', \
        'l_elbow_flex_joint','l_forearm_roll_joint','l_wrist_flex_joint','l_wrist_roll_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    # intial config
    q_arr = np.zeros((1, len(joint_idx)))
    set_joint_positions_np(robot, joint_idx, q_arr)
    # list of example targets
    targets = [[-0.15070158,  0.47726995, 1.56714123],
               [-0.36535318,  0.11249,    1.08326675],
               [-0.56491217,  0.011443,   1.2922572 ],
               [-1.07012697,  0.81909669, 0.47344636],
               [-1.11050811,  0.97000718,  1.31087581]]
    # define joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robot, joint_idx[i]).jointLowerLimit, get_joint_info(robot, joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}
    q = np.zeros((1, len(joint_names))) # start at this configuration
    target = targets[test_idx]
    # draw a blue sphere at the target
    draw_sphere_marker(target, 0.05, (0, 0, 1, 1))
    
    ### YOUR CODE HERE ###
    joint_idx = np.array(joint_idx)
    joint_limits_arr = np.array([joint_limits[joint_names[i]] for i in range(len(joint_names))])
    for i in range(len(joint_limits_arr)):
        # NOTE: sometimes, ub is lower than lb; in such a case, the joint has no limits?
        lb, ub = joint_limits_arr[i]
        if ub < lb:
            joint_limits_arr[i] = -np.pi, np.pi

    target = np.array(target)
    J = get_translation_jacobian(robot, joint_idx)
    get_jacobian_pinv(J)

    threshold = 1e-3
    alpha = 1e-3
    q_arr[-1] = 0.2
    while True:
        # NOTE: get_ee_transform implicitly sets config to joint_vals
        cur = get_ee_transform(robot, joint_idx, joint_vals=q_arr[0])[:3,3]
        xdot = target - cur
        error = np.linalg.norm(xdot)
        if error < threshold:
            break

        J = get_translation_jacobian(robot, joint_idx)
        J_pinv = get_jacobian_pinv(J)
        qdot = J_pinv @ xdot
        qdot_norm = np.linalg.norm(qdot)
        if (qdot_norm > alpha):
            qdot = alpha * (qdot / qdot_norm)
        q_arr += qdot

        # non-vectorized solution
        # for i in range(len(joint_idx)):
        #     jidx = joint_idx[i]
        #     jname = joint_names[i]
        #     lb, ub = joint_limits[jname]
        #     q_i = q_arr[0,i]
        #     q_arr[0,i] = max(q_i, lb) if q_i < lb else min(q_i, ub)

        # BUG: Why are some of the joint_limit lowerbounds
        # greater than the corresponding upperbounds?
        # We are not handling this correctly.
        below = q_arr[0,:] < joint_limits_arr[:,0]
        q_arr[0,below] = np.maximum(q_arr[0,below], joint_limits_arr[below,0])
        q_arr[0,~below] = np.minimum(q_arr[0,~below], joint_limits_arr[~below,1])



    ### YOUR CODE HERE ###

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
