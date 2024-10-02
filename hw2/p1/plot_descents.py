import numpy as np
import matplotlib.pyplot as plt
from gradientdescent import grad_descend
from newtonsmethod import newtons_method

f = lambda x : np.exp(0.5 * x + 1) + np.exp(-0.5 * x - 0.5) + 5 * x
fp = lambda x : np.exp(0.5 * x + 1)/2 - np.exp(-0.5 * x - 0.5)/2 + 5
fpp = lambda x : np.exp(0.5 * x + 1)/4 + np.exp(-0.5 * x - 0.5)/4

xs_gd = grad_descend(f, fp, 5)
xs_nt = newtons_method(f, fp, fpp, 5)
fig, ax = plt.subplots()

"plot for i."
x = np.linspace(-10, 10, 1000)
ax.set_title("Plot i.")
ax.set_xlabel("x (optimization variable)")
ax.set_ylabel("y (objective value)")
ax.plot(x, f(x), 'k-')
ax.plot(xs_gd, f(xs_gd), '-ro')
ax.plot(xs_nt, f(xs_nt), '-mx')

"plot for ii."
fig2, ax2 = plt.subplots()
len_gd = len(xs_gd)
len_nt = len(xs_nt)
print("num_iter(gd):", len_gd)
print("num_iter(nt):", len_nt)
ivals = np.arange(max(len_gd, len_nt))
ax2.set_title("Plot ii.")
ax2.set_xlabel("i (iteration number)")
ax2.set_ylabel("f(x_i) (objective value at ith iteration)")
ax2.set_xticks(range(max(len_gd, len_nt) + 1))
ax2.plot(ivals[:len_gd], f(xs_gd), '-ro')
ax2.plot(ivals[:len_nt], f(xs_nt), '-mx')

plt.show()
