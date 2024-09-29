import backtracking

def newtons_method(f, fp, fpp, x):
    epsilon = 0.0001
    def lambda_sq(x):
        rv = (fp(x) ** 2) / fpp(x)
        '''
        confused; how can we guarantee rv is non-negative?

        This is a concern because:
        On the slides, the stopping criterion is:
        lambda ^ 2 / 2 ≤ epsilon

        Instead of something like:
        |lambda ^ 2 / 2| ≤ epsilon

        I guess a better question is how can we guarantee that
        lambda (which is a square root over some term) is 
        even well-defined? What if the term inside the square
        root is negative?
        '''
        assert(rv >= 0)
        return rv

    rv = [x]
    while (lambda_sq(x) / 2 > epsilon):
        dx = -fp(x) / fpp(x)
        t = backtracking.bls(f, fp, x, dx)
        x += t * dx
        rv.append(x)

    return rv

