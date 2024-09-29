import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("calibration.txt")

print("\nQ9. c)")
bases = [np.vectorize(f) for f in [
    lambda x : 1,
    lambda x : x,
    lambda x : np.maximum(0, x - (-0.5)),
    lambda x : np.maximum(0, x - 0.5),
]]

mat_A = np.column_stack([f(data[:,0]) for f in bases])
pinv_A = np.linalg.pinv(mat_A)
params = np.matmul(pinv_A, data[:,1])

def piecewise(x):
    return (1 * params[0] + 
            x * params[1] + 
            np.maximum(0, x - (-0.5)) * params[2] + 
            np.maximum(0, x - 0.5) * params[3])
sq_errors = (piecewise(data[:,0]) - data[:,1]) ** 2

print("piecewise params:\n", params)
print("SSE =", sum(sq_errors))
print("pred(0.68) =", piecewise(0.68))

fig2, ax2 = plt.subplots()
ax2.set_title("Piece-wise linear least-squares fit")
ax2.scatter(data[:,0], data[:,1], color='blue', marker='x')
ax2.set_ylabel("Measured position")
ax2.set_xlabel("Commanded position")
sample_x = np.array([-1, -0.5, 0.5, 1])
sample_y = piecewise(sample_x)
ax2.plot(sample_x, sample_y, 'r-') 
plt.show()



