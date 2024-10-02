import path
import numpy as np
from sgd import sgd
from gradientdescent import grad_descend
from newtonsmethod import newtons_method
from SGDtest import fi, fiprime, fsum, fsumprime, fsumprimeprime, maxi
import time

if __name__ == "__main__":
    x0 = -5

    start = time.time()
    fsum_xs_gd = np.array(fsum(grad_descend(fsum, fsumprime, x0)))
    gd_time = time.time() - start

    start = time.time()
    fsum_xs_nt = np.array(fsum(newtons_method(fsum, fsumprime, fsumprimeprime, x0)))
    nt_time = time.time() - start

    num_iter = 1000
    num_runs = 30
    start = time.time()
    fsum_xs_sgd = np.array([fsum(sgd(fi, fiprime, maxi, num_iter, x0)[-1]) for _ in range(num_runs)])
    sgd_time = time.time() - start


    print('''\
Gradient Descent (1 run):
    fsum_x = %.3f
    num_iter = %d
    runtime = %.3f\
    ''' % (fsum_xs_gd[-1], len(fsum_xs_gd), gd_time)
    )

    print('''
Newton's Method (1 run):
    fsum_x = %.3f
    num_iter = %d
    runtime = %.3f\
    ''' % (fsum_xs_nt[-1], len(fsum_xs_nt), nt_time)
    )

    print('''
SGD (%d runs):
    fsum_x var = %.3f
    fsum_x mean = %.3f
    num_iter = %d
    av. runtime = %.3f\
    ''' % (num_runs, np.var(fsum_xs_sgd), np.mean(fsum_xs_sgd), num_iter, sgd_time / num_runs)
    )
    # TODO: I think var and mean is supposed to be lower for 1k iterations, but this doesn't
    # seem consistently true. Maybe due to large t? A problem?
    print('''
Note:
- SGD was run 30 times; average and variance were taken
- GD and NT were run only once each since they are deterministic.

i)
runtime: SGD < NT < GD

Although SGD has many more iterations than NT and GD, each iteration is
many orders of magnitudes cheaper because it only requires differentiating a single fi function,
rather than the entire fsum function as in NT and GD.

NT is faster than GD because it considers the 2nd derivative (in addition to the 1st) to 
determine the next step. This has two effects:
1. Faster convergence: Each update is more precise, bringing the {x_i} sequence
closer to local optimum with minimal overshooting. This means fewer iterations
needed for convergence.
2. Cheaper iterations: Since the update direction dx incorporates 2nd derivative curvature
information, the steps are better tuned. Thus, per iteration, fewer iterations of backtracking 
line search is required to choose an appropriate step-size t.

ii)
GD and NT have near-identical fsum values. 
This is an inevitable consequence of the epsilon-based stopping condition,
which guarantees that the final fsum value is extremely close to local optimum.

Furthermore, SGD, despite it being an approximate, stochastic method, (unlike
the exact NT and GD algorithms) has an (average) fsum value that is very 
close to that of GD and NT. This shows that SGD was able to converge sufficiently 
to local optimum within 1000 iterations.
    ''')

