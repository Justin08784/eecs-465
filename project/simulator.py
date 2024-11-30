import numpy as np


# TODO: Replace this simulation loop with Runge-Katta.
def simulate(controls, cur_s, cur_v, states, num_states, dt):
    '''
    cur_s, cur_v, cur_u: start state, velocity, and control (accel)
    states: np.array to fill
    n: number of time steps, INCLUDING given cur_s,... (which is index 0)
    dt: time step size
    '''
    # num_controls = controls.shape[0]
    # for c in range(num_controls):
    states[0,0,:4] = cur_s
    cur_u = controls[4]

    for i in range(1, num_states):
        cos = np.cos(cur_s[3])
        sin = np.sin(cur_s[3])

        cur_s[0] += dt * (cur_v[0] * cos - cur_v[1] * sin)
        cur_s[1] += dt * (cur_v[0] * sin + cur_v[1] * cos)
        # ignore z. we will not ever update it
        cur_s[3] += dt * cur_v[3]
        states[0,i,:4] = cur_s

        cur_v[0] += dt * (cur_u[0] * cos - cur_u[1] * sin)
        cur_v[1] += dt * (cur_u[0] * sin + cur_u[1] * cos)
        # ignore z. we will not ever update it
        cur_v[3] += dt * cur_u[2]
