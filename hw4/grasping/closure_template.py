import argparse

from world import World
from utils import NUM_FRICTION_CONE_VECTORS, process_contact_point, get_f0_pre_rotation

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

def calculate_wrenches(contact_points, world):
    """
    Calculate the 6D wrenches for each contact point.
    :param contact_points: A list of contact points. Each contact point is a tuple of information from p.getContactPoints.
    :param world: The world object.
    :return: A numpy array of shape (NUM_FRICTION_CONE_VECTORS*n, 6) where n is the number of contact points.
    """
    
    #Get the object position and friction coefficient
    position, mu, max_radius= world.get_object_info()
    wrenches = []

    for contact_point in contact_points:
        contact_pos, contact_force_vector, tangent_dir = process_contact_point(contact_point)
        cone_edges = calculate_friction_cone(contact_force_vector, tangent_dir, mu)

        if cone_edges is not None:
            for i in range(NUM_FRICTION_CONE_VECTORS):
                wrench = np.zeros(6)
                cone_vector = cone_edges[i]
                wrench[:3] = cone_vector

                #Calculate the radius vector
                radius = contact_pos - position

                #Calculate the torque
                torque = np.cross(radius, cone_vector)
                #Scale the torque by the max radius so that the units are consistent with the force
                torque /= max_radius
                wrench[3:] = torque
                wrenches.append(wrench)

    wrenches = np.array(wrenches)
    return wrenches

def calculate_friction_cone(contact_force_vector, tangent_dir, mu, num_cone_vectors=NUM_FRICTION_CONE_VECTORS):
    """
    Calculate the friction cone vectors for a single contact point.
    :param contact_force_vector: The contact force vector.
    :param tangent_dir: A vector tangent to the contact surface.
    :param mu: The coefficient of friction.
    :return: A numpy array of shape (NUM_FRICTION_CONE_VECTORS, 3) where each row is a vector in the friction cone.
    """
    
    cone_edges = None
    _, contact_unit_normal, f_0_pre_rotation = get_f0_pre_rotation(contact_force_vector, mu)
    ### YOUR CODE HERE ###
    u_normal = contact_unit_normal

    origin = np.array([0,0,0])
    f0 = f_0_pre_rotation
    def get_uvec_plane(nv):
        if nv[0] == 0 and nv[1] == 0:
            return np.array([1, 0, 0])
        else:
            norm = nv[1]**2 + nv[0]**2
            return np.array([nv[1], -nv[0], 0])/norm

    import time
    start = time.time()
    p0 = get_uvec_plane(u_normal)

    about_normal = Rotation.from_rotvec(
        np.outer(np.linspace(0, 2 * np.pi, num_cone_vectors, endpoint=False), u_normal)
    )
    plane_axes = about_normal.apply(p0)

    theta = np.arctan(mu)
    about_plane_axes = Rotation.from_rotvec(theta * plane_axes)
    cone_edges = about_plane_axes.apply(f0)

    ######################

    return cone_edges

def compare_discretization(contact_point, world):
    """
    Calculate the volume of the friction cone using different discretizations.
    :param contact_point: A contact point. This is a tuple of information from p.getContactPoints.
    :param world: The world object.
    :return: None
    """
    _, mu, _= world.get_object_info()
    _, contact_force_vector, tangent_dir = process_contact_point(contact_point)

    four_vector = calculate_friction_cone(contact_force_vector, tangent_dir, mu, num_cone_vectors=4)
    eight_vector = calculate_friction_cone(contact_force_vector, tangent_dir, mu, num_cone_vectors=8)

    if four_vector is None or eight_vector is None:
        print('calculate_friction_cone not implemented')
    
    true_volume = None
    four_vector_volume = None
    eight_vector_volume = None

    ### YOUR CODE HERE ###
    theta = np.arctan(mu)
    cf_vlen = np.linalg.norm(contact_force_vector)
    base_radius = cf_vlen / np.sin(theta)
    height = cf_vlen

    true_volume = (1/3)*np.pi*(base_radius**2)*height

    phi_4 = 2*np.pi/4
    base_area_4 = 4*(1/2)*(base_radius**2)*np.sin(phi_4)
    four_vector_volume = (1/3)*base_area_4*height

    phi_8 = 2*np.pi/8
    base_area_8 = 8*(1/2)*(base_radius**2)*np.sin(phi_8)
    eight_vector_volume = (1/3)*base_area_8*height

    ######################

    print('True volume:', np.round(true_volume, 4) if true_volume is not None else 'Not implemented')
    print('4 edge volume:', np.round(four_vector_volume, 4) if four_vector_volume is not None else 'Not implemented')
    print('8 edge volume:', np.round(eight_vector_volume, 4) if eight_vector_volume is not None else 'Not implemented')
    print()

def convex_hull(wrenches):
    """
    Given a set of wrenches, determine if the object is in force closure using a convex hull.
    :param wrenches: A numpy array of shape (NUM_FRICTION_CONE_VECTORS*n, 6) where n is the number of contact points.
    :return: None
    """

    if len(wrenches) == 0:
        print('No wrench input. Is calculate_friction_cone implemented?')

    #Convex Hull
    hull = None
    #Force closure boolean
    force_closure_bool = False
    #Radius of largest hypersphere contained within convex hull
    max_radius = 0.0

    ### YOUR CODE HERE ###

    hull = ConvexHull(wrenches, qhull_options="QJ")
    tolerance = 1e-6
    offsets = hull.equations[:,-1]
    force_closure_bool = np.all(offsets <= 0)
    if force_closure_bool:
        # print("wrench_shape!!", wrenches.shape[1])
        plane_dists = np.abs(offsets / np.linalg.norm(hull.equations[:,:-1], axis=1))
        max_radius = np.min(plane_dists)
    ######################

    if hull is None:
        print('convex_hull not implemented')
    else:
        if force_closure_bool:
            print('In force closure. Maximum radius:', np.round(max_radius, 4))
        else:
            print('Not in force closure')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--g1', action='store_true', help='Run grasp 1')
    parser.add_argument('--g2', action='store_true', help='Run grasp 2')
    parser.add_argument('--custom', nargs='+', help='Run a custom grasp. Input is a list of 4 numbers: x, y, z, theta')
    args = parser.parse_args()

    world = World()
    
    print('\n\n\n========================================\n')
    input('Environment initialized. Press <ENTER> to execute grasp.')

    if args.g1:
        contact_points = world.grasp([.0, .08, 0.03, 0])
        print('\n========================================\n')
        input(f'Grasp 1 complete. Press <ENTER> to compute friction cone volumes and force closure.\n')
        print(f'Grasp 1 Contact point 1 Volumes:')
        compare_discretization(contact_points[0], world)

    if args.g2:
        contact_points = world.grasp([.0, .02, .01, 0])
        print('\n========================================\n')
        input(f'Grasp 2 complete. Press <ENTER> to compute force closure.\n')
    
    if args.custom:
        x, y, z, theta = args.custom
        contact_points = world.grasp([float(x), float(y), float(z), float(theta)])
        print('\n========================================\n')
        input(f'Custom grasp complete. Press <ENTER> to compute force closure.')

    wrenches = calculate_wrenches(contact_points, world)
    convex_hull(wrenches)
    print('\n\n')

if __name__ == '__main__':
    main()
