import numpy as np
import constants as con


# # TODO: Replace this simulation loop with Runge-Katta.
# def abs_cutoff(x, lim):
#     if x < -lim:
#         return -lim
#     if x > lim:
#         return lim
#     return x
# def simulate(controls, orig_s, orig_v, states, num_states, dt=con.dt):
#     '''
#     cur_s, cur_v, cur_u: start state, velocity, and control (accel)
#     states: np.array to fill
#     n: number of time steps, EXCLUDING given cur_s,...
#         states[c,i] will the 1st time step AFTER applying 1 dt of control 'c'
#     dt: time step size
#     '''
#     num_controls = controls.shape[0]
#     # states[:,0,:4] = orig_s
#     # states[:,0,4:] = orig_v
# 
#     cur_s = np.zeros(4)
#     cur_v = np.zeros(4)
#     for c in range(num_controls):
#         cur_s[:] = orig_s
#         cur_v[:] = orig_v
#         cur_u = controls[c]
# 
#         for i in range(0, num_states):
#             cos = np.cos(cur_s[3])
#             sin = np.sin(cur_s[3])
# 
#             cur_s[0] += dt * (cur_v[0] * cos - cur_v[1] * sin)
#             cur_s[1] += dt * (cur_v[0] * sin + cur_v[1] * cos)
#             # ignore z. we will not ever update it
#             cur_s[3] += dt * cur_v[3]
#             states[c,i,:4] = cur_s
# 
#             cur_v[0] += dt * (cur_u[0] * cos - cur_u[1] * sin)
#             cur_v[1] += dt * (cur_u[0] * sin + cur_u[1] * cos)
#             # ignore z. we will not ever update it
#             nrm = np.linalg.norm(cur_v[:3])
#             mag = con.MAX_LIN_VEL/nrm if nrm > con.MAX_LIN_VEL else 1
#             cur_v[0] *= mag
#             cur_v[1] *= mag 
#             cur_v[3] += dt * cur_u[2]
#             cur_v[3] = abs_cutoff(cur_v[3], con.MAX_ANG_VEL)
#             states[c,i,4:] = cur_v

def abs_cutoff_vec(x, lim):
    """
    Vectorized version of abs_cutoff function.
    Clamps values in x to the range [-lim, lim].
    """
    return np.clip(x, -lim, lim)

    # def simulate(controls, orig_s, orig_v, states, num_states, dt=con.dt):
    #     """
    #     Vectorized version of the simulate function without using cur_s and cur_v.
    # 
    #     Parameters:
    #         controls: np.ndarray of shape (num_controls, 3), control accelerations for each control.
    #         orig_s: np.ndarray of shape (4,), the initial state.
    #         orig_v: np.ndarray of shape (4,), the initial velocity.
    #         states: np.ndarray of shape (num_controls, num_states, 8), array to fill with state and velocity information.
    #         num_states: int, number of time steps to simulate.
    #         dt: float, time step size.
    #     """
    #     num_controls = controls.shape[0]
    # 
    #     for i in range(num_states):
    #         if i == 0:
    #             # Use orig_s and orig_v as the previous state and velocity
    #             prev_s = np.tile(orig_s, (num_controls, 1))  # Shape: (num_controls, 4)
    #             prev_v = np.tile(orig_v, (num_controls, 1))  # Shape: (num_controls, 4)
    #         else:
    #             # Use the state and velocity from the previous time step
    #             prev_s = states[:, i - 1, :4]
    #             prev_v = states[:, i - 1, 4:]
    # 
    #         # Precompute cos and sin of theta for efficiency
    #         cos_theta = np.cos(prev_s[:, 3])  # Shape: (num_controls,)
    #         sin_theta = np.sin(prev_s[:, 3])  # Shape: (num_controls,)
    # 
    #         # Update position based on previous velocity
    #         states[:, i, 0] = prev_s[:, 0] + dt * (prev_v[:, 0] * cos_theta - prev_v[:, 1] * sin_theta)  # x
    #         states[:, i, 1] = prev_s[:, 1] + dt * (prev_v[:, 0] * sin_theta + prev_v[:, 1] * cos_theta)  # y
    #         states[:, i, 2] = prev_s[:, 2]  # z coordinate (assuming no change)
    #         states[:, i, 3] = prev_s[:, 3] + dt * prev_v[:, 3]  # angle theta
    # 
    #         # Update velocity based on controls
    #         # Linear velocities
    #         states[:, i, 4] = prev_v[:, 0] + dt * (controls[:, 0] * cos_theta - controls[:, 1] * sin_theta)
    #         states[:, i, 5] = prev_v[:, 1] + dt * (controls[:, 0] * sin_theta + controls[:, 1] * cos_theta)
    #         states[:, i, 6] = prev_v[:, 2]  # z velocity (assuming no change)
    # 
    #         # Normalize linear velocity to enforce MAX_LIN_VEL
    #         nrm = np.linalg.norm(states[:, i, 4:7], axis=1, keepdims=True)  # Shape: (num_controls, 1)
    #         mag = np.where(nrm > con.MAX_LIN_VEL, con.MAX_LIN_VEL / nrm, 1)  # Scaling factor
    #         states[:, i, 4] *= mag[:, 0]  # Scale velocity x
    #         states[:, i, 5] *= mag[:, 0]  # Scale velocity y
    # 
    #         # Update angular velocity
    #         states[:, i, 7] = prev_v[:, 3] + dt * controls[:, 2]  # angular velocity
    #         states[:, i, 7] = abs_cutoff_vec(states[:, i, 7], con.MAX_ANG_VEL)  # Clamp angular velocity

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
