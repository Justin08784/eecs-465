import numpy as np


# TODO: Replace this simulation loop with Runge-Katta.
def simulate(controls, orig_s, orig_v, states, num_states, dt):
    '''
    cur_s, cur_v, cur_u: start state, velocity, and control (accel)
    states: np.array to fill
    n: number of time steps, EXCLUDING given cur_s,...
        states[c,i] will the 1st time step AFTER applying 1 dt of control 'c'
    dt: time step size
    '''
    num_controls = controls.shape[0]
    # states[:,0,:4] = orig_s
    # states[:,0,4:] = orig_v

    cur_s = np.zeros(4)
    cur_v = np.zeros(4)
    for c in range(num_controls):
        cur_s[:] = orig_s
        cur_v[:] = orig_v
        cur_u = controls[c]

        for i in range(0, num_states):
            cos = np.cos(cur_s[3])
            sin = np.sin(cur_s[3])

            cur_s[0] += dt * (cur_v[0] * cos - cur_v[1] * sin)
            cur_s[1] += dt * (cur_v[0] * sin + cur_v[1] * cos)
            # ignore z. we will not ever update it
            cur_s[3] += dt * cur_v[3]
            states[c,i,:4] = cur_s

            cur_v[0] += dt * (cur_u[0] * cos - cur_u[1] * sin)
            cur_v[1] += dt * (cur_u[0] * sin + cur_u[1] * cos)
            # ignore z. we will not ever update it
            cur_v[3] += dt * cur_u[2]
            states[c,i,4:] = cur_v
