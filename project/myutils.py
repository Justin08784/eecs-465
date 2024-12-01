import json
from pybullet_tools.parse_json import parse_robot, parse_body
from pybullet_tools.utils import set_joint_positions, \
    wait_if_gui, wait_for_duration

import pybullet as p
from pybullet_tools.utils import cached_fn, get_buffered_aabb, pairwise_collision
from pybullet_tools.utils import aabb_overlap, set_pose
from itertools import product
MAX_DISTANCE = 0. # 0. | 1e-3

# def load_env(env_file):
#     # load robot and obstacles defined in a json file
#     with open(env_file, 'r') as f:
#         env_json = json.loads(f.read())
#     robots = {robot['name']: parse_robot(robot) for robot in env_json['robots']}
#     bodies = {body['name']: parse_body(body) for body in env_json['bodies']}
#     return robots, bodies

def get_collision_fn(body, obstacles=[], use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
    '''
    get collision_fn for a body without links or attachments
    '''
    get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)

    def collision_fn(s, verbose=False):
        get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)
        set_pose(body, s)

        for obst in obstacles:
            if (not use_aabb or aabb_overlap(get_moving_aabb(body), get_obstacle_aabb(obst))) \
                    and pairwise_collision(body, obst, **kwargs):
                #print(get_body_name(body1), get_body_name(body2))
                if verbose: print(body, obst)
                return True
        return False
    return collision_fn

# def get_collision_fn_PR2(robot, joints, obstacles):
#     # check robot collision with environment
#     disabled_collisions = get_disabled_collisions(robot)
#     return get_collision_fn(robot, joints, obstacles=obstacles, attachments=[], \
#         self_collisions=True, disabled_collisions=disabled_collisions)
# 
# def execute_trajectory(robot, joints, path, sleep=None):
#     # Move the robot according to a given path
#     if path is None:
#         print('Path is empty')
#         return
#     print('Executing trajectory')
#     for bq in path:
#         set_joint_positions(robot, joints, bq)
#         if sleep is None:
#             wait_if_gui('Continue?')
#         else:
#             wait_for_duration(sleep)
#     print('Finished')

# def draw_sphere_marker(position, radius, color):
#    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
#    marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
#    return marker_id
# 
# 
# def draw_line(start, end, width, color):
#     line_id = p.addUserDebugLine(start, end, color, width)
#     return line_id
