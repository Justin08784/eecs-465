import path
import numpy as np
import matplotlib.pyplot as plt
from sgd import sgd
from SGDtest import fi, fiprime, fsum, maxi

if __name__ == "__main__":
    num_iter = 1000

    fig, ax = plt.subplots()
    x0 = -5
    xs = sgd(fi, fiprime, maxi, num_iter, x0)
    fsum_xs = fsum(xs)
    diffs = np.diff(fsum_xs)
    ax.set_title("Plot 2.b)")
    ax.set_xticks(np.arange(0, num_iter + 1, 100))
    ax.set_xlabel("i (iteration number)")
    ax.set_ylabel("fsum(x_i) (objective value at iteration i)")
    ax.plot(np.arange(num_iter + 1), fsum_xs, 'r-')

    is_strictly_decreasing = np.all(diffs < 0)
    print('''\
Is fsum(x_i) strictly decreasing?: {0}

Unlike gradient descent, which computes the exact gradient over fsum
–– and thus over all of the cost functions (i.e. the fi's)–– to determine its next step, 
SGD approximates the gradient by computing it over a random subset
of the cost functions (in our case just 1 fi at a time).

The gradient provides the direction of steepest descent. Since SGD steps using 
an approximated gradient (based on one fi), each step is noisy and may not 
necessarily follow the direction of steepest descent for fsum. As a result, any 
individual update can cause an increase (or non-decrease) in fsum, even though 
the overall trend tends to minimize fsum in the long run.
    '''.format(is_strictly_decreasing))

    plt.show()
