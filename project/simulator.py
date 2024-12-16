import numpy as np
import constants as con

def abs_cutoff_vec(x, lim):
    """
    Vectorized version of abs_cutoff function.
    Clamps values in x to the range [-lim, lim].
    """
    return np.clip(x, -lim, lim)

def simulate(controls, orig_s, orig_v, states, num_states, dt=con.dt):
    """
    Vectorized version of the simulate function.

    Parameters:
        controls: np.ndarray of shape (num_controls, 3), control accelerations for each control.
        orig_s: np.ndarray of shape (4,), the initial state.
        orig_v: np.ndarray of shape (4,), the initial velocity.
        states: np.ndarray of shape (num_controls, num_states, 8), array to fill with state and velocity information.
        num_states: int, number of time steps to simulate.
        dt: float, time step size.
    """
    num_controls = controls.shape[0]

    # Initialize cur_s and cur_v for each control
    cur_s = np.tile(orig_s, (num_controls, 1))  # Shape: (num_controls, 4)
    cur_v = np.tile(orig_v, (num_controls, 1))  # Shape: (num_controls, 4)

    # Precompute cos and sin for efficiency
    cos_theta = np.cos(cur_s[:, 3])  # Shape: (num_controls,)
    sin_theta = np.sin(cur_s[:, 3])  # Shape: (num_controls,)

    for i in range(num_states):
        # Update position based on current velocity
        cur_s[:, 0] += dt * (cur_v[:, 0] * cos_theta - cur_v[:, 1] * sin_theta)  # Update x
        cur_s[:, 1] += dt * (cur_v[:, 0] * sin_theta + cur_v[:, 1] * cos_theta)  # Update y
        cur_s[:, 3] += dt * cur_v[:, 3]  # Update angle theta

        # Store current state (position/orientation) in the states array
        states[:, i, :4] = cur_s

        # Update velocity based on controls
        cur_v[:, 0] += dt * (controls[:, 0] * cos_theta - controls[:, 1] * sin_theta)  # Linear velocity x
        cur_v[:, 1] += dt * (controls[:, 0] * sin_theta + controls[:, 1] * cos_theta)  # Linear velocity y

        # Normalize linear velocity to enforce MAX_LIN_VEL
        nrm = np.linalg.norm(cur_v[:, :3], axis=1, keepdims=True)  # Shape: (num_controls, 1)
        mag = np.where(nrm > con.MAX_LIN_VEL, con.MAX_LIN_VEL / nrm, 1)  # Scaling factor
        cur_v[:, :2] *= mag  # Scale velocities x, y

        # Update angular velocity
        cur_v[:, 3] += dt * controls[:, 2]  # Update angular velocity
        cur_v[:, 3] = abs_cutoff_vec(cur_v[:, 3], con.MAX_ANG_VEL)  # Clamp angular velocity

        # Store current velocity in the states array
        states[:, i, 4:] = cur_v

        # Recompute cos and sin for updated orientation
        cos_theta = np.cos(cur_s[:, 3])
        sin_theta = np.sin(cur_s[:, 3])
