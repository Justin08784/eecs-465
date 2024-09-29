import backtracking

'''
fp = derivative of f
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

    return rv

