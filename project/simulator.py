import numpy as np


def simulate(states, cur_s, cur_v, cur_u, n, dt):
    '''
    cur_s, cur_v, cur_u: start state, velocity, and control (accel)
    states: np.array to fill
    n: number of time steps; cur_s,... is index 0
    dt: time step size
    '''
    states[0] = cur_s
    for i in range(1, n):
        cur_s += cur_v * dt
        states[i] = cur_s
        cur_v += cur_u * dt

