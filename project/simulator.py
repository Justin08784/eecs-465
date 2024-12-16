import numpy as np
import constants as con

def abs_cutoff_vec(x, lim):
    """
    Clamps values in x to the range [-lim, lim].
    """
    return np.clip(x, -lim, lim)

def simulate(controls, orig_s, orig_v, states, num_states, dt=con.dt):
    """
    Vectorized version of the simulate function.

    Parameters:
        controls: array of shape (num_controls, 3), control accelerations for each control.
        orig_s: array of shape (4,), the initial state.
        orig_v: array of shape (4,), the initial velocity.
        states: array of shape (num_controls, num_states, 8), array to fill with state and velocity information.
        num_states: int, number of time steps to simulate.
        dt: float, time step size.
    """
    num_controls = controls.shape[0]

    # initialize cur_s and cur_v for each control
    cur_s = np.tile(orig_s, (num_controls, 1))  # shape: (num_controls, 4)
    cur_v = np.tile(orig_v, (num_controls, 1))  # shape: (num_controls, 4)

    # precompute cos and sin for efficiency
    cos_theta = np.cos(cur_s[:, 3])  # shape: (num_controls,)
    sin_theta = np.sin(cur_s[:, 3])  # shape: (num_controls,)

    for i in range(num_states):
        # update position based on current velocity
        cur_s[:, 0] += dt * (cur_v[:, 0] * cos_theta - cur_v[:, 1] * sin_theta)  # update x
        cur_s[:, 1] += dt * (cur_v[:, 0] * sin_theta + cur_v[:, 1] * cos_theta)  # update y
        cur_s[:, 3] += dt * cur_v[:, 3]  # update angle theta

        # store current state (position/orientation) in the states array
        states[:, i, :4] = cur_s

        # update velocity based on controls
        cur_v[:, 0] += dt * (controls[:, 0] * cos_theta - controls[:, 1] * sin_theta)  # linear velocity x
        cur_v[:, 1] += dt * (controls[:, 0] * sin_theta + controls[:, 1] * cos_theta)  # linear velocity y

        # normalize linear velocity to enforce MAX_LIN_VEL
        nrm = np.linalg.norm(cur_v[:, :3], axis=1, keepdims=True)  # shape: (num_controls, 1)
        mag = np.where(nrm > con.MAX_LIN_VEL, con.MAX_LIN_VEL / nrm, 1)  # scaling factor
        cur_v[:, :2] *= mag  # scale velocities x, y

        # update angular velocity
        cur_v[:, 3] += dt * controls[:, 2]  # update angular velocity
        cur_v[:, 3] = abs_cutoff_vec(cur_v[:, 3], con.MAX_ANG_VEL)  # clamp angular velocity

        # store current velocity in the states array
        states[:, i, 4:] = cur_v

        # recompute cos and sin for updated orientation
        cos_theta = np.cos(cur_s[:, 3])
        sin_theta = np.sin(cur_s[:, 3])
