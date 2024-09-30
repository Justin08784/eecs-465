import path
import numpy as np
from sgd import sgd
from SGDtest import fi, fiprime, fsum, maxi

if __name__ == "__main__":
    # TODO: It asks us to compute fsum(x∗), but what is this exactly?
    x0 = -5
    num_runs = 30

    num_iter = 750
    fsum_xs_iter750 = np.array([fsum(sgd(fi, fiprime, maxi, num_iter, x0)[-1]) for _ in range(num_runs)])
    
    num_iter = 1000
    fsum_xs_iter1k = np.array([fsum(sgd(fi, fiprime, maxi, num_iter, x0)[-1]) for _ in range(num_runs)])


    print('''\
750 iterations:
    var = %.3f
    mean = %.3f\
    ''' % (np.var(fsum_xs_iter750), np.mean(fsum_xs_iter750))
    )

    print('''\
1000 iterations:
    var = %.3f
    mean = %.3f\
    ''' % (np.var(fsum_xs_iter1k), np.mean(fsum_xs_iter1k))
    )

    # TODO: I think var and mean is supposed to be lower for 1k iterations, but this doesn't
    # seem consistently true. Maybe due to large t? A problem?
    print('''\
The 30-run mean is virtually identical for both iteration counts.
The 30-run variances fluctuate: sometimes higher for 750 iterations, sometimes lower.

This suggests that by 750 iterations, SGD has likely converged to a local optimum.
Additional iterations don't reduce the objective function further but instead cause the 
algorithm to "wander" around the optimum.

Interestingly, if we lower t to t = 0.5, then both 30-run mean and variance is lower for
1k iterations. With the reduced step size, 750 iterations may not be sufficient for convergence.
This could explain the less-optimal mean for 750 iterations. Additionally, if SGD is 
still making relatively large steps at iteration 750, small differences in the stochastic 
updates could cause noticeable differences in the final function value, explaining 
the larger variance for 750 iterations.''')

