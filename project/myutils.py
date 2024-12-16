import json
from pybullet_tools.parse_json import parse_robot, parse_body
from pybullet_tools.utils import set_joint_positions, \
    wait_if_gui, wait_for_duration

import pybullet as p
from pybullet_tools.utils import cached_fn, get_buffered_aabb, pairwise_collision
from pybullet_tools.utils import aabb_overlap, set_pose
from itertools import product
MAX_DISTANCE = 0. # 0. | 1e-3

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

