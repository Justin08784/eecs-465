import numpy as np
import path

'''
stochastic gradient descent

params:
- fi  : maps 'n' to 'nth' component objective function
  i.e. n ↦ f_n
- fip : maps 'n' to derivative of 'nth' comp. objective function
  i.e. n ↦ f_n'
- maxi = number of comp. objective functions f_i's
- num_iter = iteration limit (stopping condition)
- x = initial value of optimization variable
- t = step-size

returns: np.array([x_0, x_1,..., x_n])
- x_i = x at iteration i
- x_n = a (local) optimum
'''
def sgd(fi, fip, maxi, num_iter, x, t = 1):
    fn_idxs = np.random.choice(maxi, num_iter)

    rv = [x]
    for i in range(num_iter):
        dx = -fip(x, fn_idxs[i])
        x += t * dx
        rv.append(x)

    return np.array(rv)

