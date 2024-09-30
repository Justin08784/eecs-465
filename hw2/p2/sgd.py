import numpy as np
import path

'''
fp = derivative of f
'''
def sgd(fi, fip, maxi, num_iter, x):
    t = 1

    fn_idxs = np.random.choice(maxi, num_iter)

    rv = [x]
    for i in range(num_iter):
        dx = -fip(x, fn_idxs[i])
        x += t * dx
        rv.append(x)

    return np.array(rv)

