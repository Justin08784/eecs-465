import backtracking
import numpy as np

'''
gradient descent

params:
- f = objective function
- fp = f'
- x = initial value of optimization variable

returns: np.array([x_0, x_1,..., x_n])
- x_i = x at iteration i
- x_n = a (local) optimum
'''
def grad_descend(f, fp, x):
    epsilon = 0.0001
    t = 1

    rv = [x]
    while abs(fp(x)) > epsilon:
        dx = -fp(x)
        t = backtracking.bls(f, fp, x, dx)
        x += t * dx
        rv.append(x)

    return np.array(rv)

