import numpy as np
import path
import os, sys
import SGDtest

'''
fp = derivative of f
'''
def sgd(fi, fip, maxi, x):
    num_iter = 1000
    t = 1

    fn_idxs = np.random.choice(maxi, num_iter)
    # print(fn_idxs)
    # exit(0)

    rv = [x]
    for i in range(num_iter):
        dx = -fip(x, fn_idxs[i])
        x += t * dx
        rv.append(x)

    return rv

if __name__ == '__main__':
    xs = np.array(
        sgd(SGDtest.fi,
            SGDtest.fiprime,
            SGDtest.maxi,
            -5))
    print(xs)
