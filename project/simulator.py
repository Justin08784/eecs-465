import numpy as np
import constants as c


# TODO: Replace this simulation loop with Runge-Katta.
def simulate(controls, orig_s, orig_v, states, num_states, dt=c.dt):
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

def extend_to(src, dst):
    found = False
    start = time.time()
    errors = np.zeros(CONTROL_SET.shape[0], dtype=np.float64)
    errors[:] = np.inf
    prev_min_error = np.inf
    curr_min_error = np.inf
    argmin = np.zeros(CONTROL_SET.shape[0], dtype=int)

    for i in range(MAX_NUM_EXTENDS):
        simulate(CONTROL_SET, curs, curv, sim_states, NUM_SIM_STEPS, dt)
        for c in range(CONTROL_SET.shape[0]):
            col_t = NUM_SIM_STEPS # first colliding time step
            for t in range(NUM_SIM_STEPS):
                # look for first collision (if any)
                pos = sim_states[c,t,:3]
                quat = p.getQuaternionFromEuler((0,0,sim_states[c,t,3]))
                if collision_fn((pos, quat)):
                    # collided
                    col_t = t
                    break

            if col_t == 0:
                # first time step is already colliding. quit
                # we should exit at "Failed!"
                continue

            # distance metric weights
            lin_w = 1
            ang_w = 0.001
            dists_sq = (
                # NOTE: if you change lin_error power from 2 to 12, it corrects
                # more aggressively at longer distances, leading to better perf
                lin_w*np.sum((sim_states[c,:col_t,:3] - tmp_sg[:3])**2, axis=1)+\
                ang_w*np.minimum(
                    abs(sim_states[c,:col_t,3] - tmp_sg[3]),
                    2*np.pi - abs(sim_states[c,:col_t,3] - tmp_sg[3])
                )**2
            )**0.5
            argmin[c] = np.argmin(dists_sq)
            errors[c] = dists_sq[argmin[c]]
            if errors[c] < epsilon:
                print("Found!")
                found = True
                break
            if found:
                break
        # print(sim_states.shape, i)

        # pick control with minimum error
        opt_ctrl = np.argmin(errors)
        # time step in trail that had minimum error
        opt_idx = argmin[opt_ctrl]
        curs[:] = sim_states[opt_ctrl, opt_idx,:4]
        curv[:] = sim_states[opt_ctrl, opt_idx,4:]
        curr_min_error = errors[opt_ctrl]
        if curr_min_error >= prev_min_error - 0.01:
            # no improvement. quit
            print("Failed!")
            break
        prev_min_error = curr_min_error

        trail_len = (opt_idx + 1)
        used_len = tree_cur + trail_len # tree_cur = current used length
        if used_len > tree_len:
            # exponentially resize tree
            expansion_factor = 2**math.ceil(math.log2((used_len) / tree_len))
            new_arr = np.zeros((state_tree.shape[0] * expansion_factor, state_tree.shape[1]))
            new_arr[:tree_cur, :] = state_tree[:tree_cur,:]
            new_arr[tree_cur:, :] = np.inf
            state_tree = new_arr
            tree_len *= expansion_factor
        # add optimal trails to state_tree
        state_tree[tree_cur:used_len] = sim_states[opt_ctrl, :trail_len, :]
        tree_cur = used_len

        if found:
            break
    print(time.time() - start)
