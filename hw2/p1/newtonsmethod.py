import backtracking
import numpy as np

'''
newtons method

params:
- f = objective function
- fp, fpp = f', f''
- x = initial value of optimization variable

returns: np.array([x_0, x_1,..., x_n])
- x_i = x at iteration i
- x_n = a (local) optimum
'''
def newtons_method(f, fp, fpp, x):
    epsilon = 0.0001
    lambda2 = lambda x : (fp(x) ** 2) / fpp(x)

    rv = [x]
    while (lambda2(x) / 2 > epsilon):
        dx = -fp(x) / fpp(x)
        t = backtracking.bls(f, fp, x, dx)
        x += t * dx
        rv.append(x)

    return np.array(rv)

