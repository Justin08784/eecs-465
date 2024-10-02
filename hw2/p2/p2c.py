import path
import numpy as np
from sgd import sgd
from HW2files.SGDtest import fi, fiprime, fsum, maxi

if __name__ == "__main__":
    x0 = -5
    num_runs = 30

    def get_fsum_xs(num_iter, t):
        return np.array([fsum(sgd(fi, fiprime, maxi, num_iter, x0, t=t)[-1]) for _ in range(num_runs)])

    fsum_xs_iter750_t1 = get_fsum_xs(num_iter=750, t=1)
    fsum_xs_iter750_tH = get_fsum_xs(num_iter=750, t=0.5)

    fsum_xs_iter1k_t1 = get_fsum_xs(num_iter=1000, t=1)
    fsum_xs_iter1k_tH = get_fsum_xs(num_iter=1000, t=0.5)


    print('''\
t = 1:
750 iterations:
    var = %.3f
    mean = %.3f
1000 iterations:
    var = %.3f
    mean = %.3f\
    ''' % (np.var(fsum_xs_iter750_t1), np.mean(fsum_xs_iter750_t1),
           np.var(fsum_xs_iter1k_t1), np.mean(fsum_xs_iter1k_t1))
    )

    print('''
t = 0.5:
750 iterations:
    var = %.3f
    mean = %.3f
1000 iterations:
    var = %.3f
    mean = %.3f\
    ''' % (np.var(fsum_xs_iter750_tH), np.mean(fsum_xs_iter750_tH),
           np.var(fsum_xs_iter1k_tH), np.mean(fsum_xs_iter1k_tH))
    )

    print('''\
The 30-run mean is virtually identical for both iteration counts.
The 30-run variances fluctuate: sometimes higher for 750 iterations, sometimes lower.

This suggests that by 750 iterations, SGD has likely converged to a local optimum.
Additional iterations don't reduce the objective function further but instead cause the 
algorithm to "wander" around the optimum.

If we lower t to t = 0.5, then both 30-run mean and variance are lower for
1k iterations. With the reduced step size, 750 iterations may not be sufficient for convergence,
explaining the less-optimal mean for 750 iterations. Additionally, if SGD is 
still making relatively large steps at iteration 750, small differences in the stochastic 
updates could cause noticeable differences in the final objective value, explaining 
the larger variance for 750 iterations.''')

