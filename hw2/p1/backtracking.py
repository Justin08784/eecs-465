'''
backtracking line search

params:
- f = objective function
- fp = f'
- x = curr value of optimization variable
- dx = delta x (step direction)

returns: t (step-size)
'''
def bls(f, fp, x, dx):
    alpha = 0.1
    beta = 0.6
    t = 1

    while f(x + t * dx) > f(x) + alpha * t * fp(x) * dx:
        t *= beta

    return t
