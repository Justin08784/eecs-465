import cvxpy as cvx
import numpy as np
import barrier



if __name__ == "__main__":
    #the optimization function c:
    c = np.array([2, 1])

    #pick a starting point (this can be done autonomously but we'll do it by hand)
    x = cvx.Variable(2)

    hyperplanes = np.array(
        [[0.7071,    0.7071, 1.5], 
        [-0.7071,    0.7071, 1.5],
        [0.7071,    -0.7071, 1],
        [-0.7071,    -0.7071, 1]]
    )

    #let's break down the data into variables we are familiar with
    a = hyperplanes[:,:2] # each column is the "a" part of a hyperplane
    b = hyperplanes[:,2] # each row is the "b" part of a hyperplane (only one element in each row)

    constraints = [a @ x <= b]
    objective = cvx.Minimize(c.T @ x)
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    print("The optimal point: (%f, %f)" % (x.value[0], x.value[1]))

