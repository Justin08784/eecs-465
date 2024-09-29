import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("calibration.txt")
mat_A = np.column_stack([np.ones(data.shape[0]), data[:,0]])

pinv_A = np.linalg.pinv(mat_A)
params = np.matmul(pinv_A, data[:,1])
b, m = params
f = lambda x : m * x + b
sq_errors = (f(data[:,0]) - data[:,1]) ** 2

print("Q9. a)")
print("line parameters:\n", params)
print("SSE =", sum(sq_errors))

fig1, ax1 = plt.subplots()
ax1.set_title("Linear least-squares fit")
ax1.scatter(data[:,0], data[:,1], color='blue', marker='x')
ax1.set_ylabel("Measured position")
ax1.set_xlabel("Commanded position")
ax1.axline((0,b), slope=m, color='red')


print("\nQ9. b)")
print('''\
This problem is overdetermined. There are 21 data-points, thus yielding 21 equations,
which is more than the 2 unknowns (slope and y-intercept parameters of the line).\
''')

plt.show()
