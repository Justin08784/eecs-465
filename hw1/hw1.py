import numpy as np
from pprint import pprint


# Q5
'''
Ax = b
'''

# Q5a)
mat_A = np.array([
    [0,0,-1],
    [4,1,1],
    [-2,2,1]
])
b = np.array([[3,1,1]]).T
mat_aug = np.c_[mat_A, b]

rank_A = np.linalg.matrix_rank(mat_A)
rank_aug = np.linalg.matrix_rank(mat_aug)
print("\nQ5. a)")
print("rank(A) = {0}, rank([A|b]) = {1}, n = {2}".format(rank_A, rank_aug, mat_A.shape[1]))
print("rank(A) = rank([A|B]) = n => x has a unique solution")

x = np.linalg.solve(mat_A, b)
print("x =\n", x)

# Q5b)
mat_A = np.array([
    [0,-2,6],
    [-4,-2,-2],
    [2,1,1]
])
b = np.array([[1,-2,0]]).T
mat_aug = np.c_[mat_A, b]

rank_A = np.linalg.matrix_rank(mat_A)
rank_aug = np.linalg.matrix_rank(mat_aug)
print("\nQ5. b)")
print("rank(A) = {0}, rank([A|b]) = {1}, n = {2}".format(rank_A, rank_aug, mat_A.shape[1]))
print("rank(A) < rank([A|B]) => the system is inconsistent (i.e. no solution)")


# Q5c)
mat_A = np.array([
    [2,-2],
    [-4,3],
])
b = np.array([[3,-2]]).T
mat_aug = np.c_[mat_A, b]

rank_A = np.linalg.matrix_rank(mat_A)
rank_aug = np.linalg.matrix_rank(mat_aug)
print("\nQ5. c)")
print("rank(A) = {0}, rank([A|b]) = {1}, n = {2}".format(rank_A, rank_aug, mat_A.shape[1]))
print("rank(A) = rank([A|B]) = n => x has a unique solution")

x = np.linalg.solve(mat_A, b)
print("x =\n", x)



'''
Q6
'''
mat_A = np.array([
    [1,2],
    [3,-1],
])
mat_B = np.array([
    [-2,-2],
    [4,-3],
])

# Q6a)
print("\nQ6. a)")
print("A + 2B =\n", mat_A + 2 * mat_B)
# Q6b)
print("\nQ6. b)")
print("AB =\n", np.matmul(mat_A, mat_B))
print("BA =\n", np.matmul(mat_B, mat_A))
# Q6c)
print("\nQ6. c)")
print("A.T =\n", mat_A.T)
# Q6d)
print("\nQ6. d)")
print("B^2 =\n", np.matmul(mat_B, mat_B))
# Q6e)
print("\nQ6. e)")
print("(A.T)(B.T) =\n", np.matmul(mat_A.T, mat_B.T))
print("(AB).T =\n", np.matmul(mat_A, mat_B).T)
# Q6f)
print("\nQ6. f)")
print("det(A) =", int(np.linalg.det(mat_A)))
# Q6g)
print("\nQ6. g)")
print("B^-1 =\n", np.linalg.inv(mat_B))


'''
Q7
'''
print("\nQ7.")
rotx = lambda theta : np.array([
        [1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta),np.cos(theta)],
])
roty = lambda theta : np.array([
    [np.cos(theta),0,np.sin(theta)],
    [0,1,0],
    [-np.sin(theta),0,np.cos(theta)],
])
rotz = lambda theta : np.array([
    [np.cos(theta),-np.sin(theta),0],
    [np.sin(theta),np.cos(theta),0],
    [0,0,1],
])

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
rot1 = rotz(np.pi/2)
rot2 = roty(-np.pi/5)
rot3 = rotz(np.pi)
net_rot = np.matmul(rot3, np.matmul(rot2, rot1))
print(net_rot)


'''
Q8
'''
point_target = np.array([-0.4,0.9,0])
pose_robot = np.array([
    [np.sqrt(2)/2,np.sqrt(2)/2,0,1.7],
    [-np.sqrt(2)/2,np.sqrt(2)/2,0,2.1],
    [0,0,1,0],
    [0,0,0,1],
])
point_robot = np.array([1.7,2.1,0]) 
print("\nQ8. a)")
v = point_target - point_robot
print("v =", v)

print("\nQ8. b)")
normalize_vector = lambda v : v / np.linalg.norm(v)

# Since robot remains upright (relative to xy plane), keep same z-axis.
new_z = np.array([0,0,1])
# Camera, i.e. x-axis, should point towards target (i.e. be aligned with v-vector).
new_x = normalize_vector(v)
# New y-axis should be perpendicular, i.e. the cross-prod., of the new z and x (follow RH rule)
new_y = normalize_vector(np.cross(new_z, new_x))
pose_dst = np.column_stack([new_x, new_y, new_z, point_robot])
pose_dst = np.vstack([pose_dst, [0,0,0,1]])
print("dst_pose =\n", pose_dst)

print("\nQ8. c)")
print('''\
A square-matrix M is a rotation matrix IFF
    1) det(M) == 1
    2) M.T == M^-1\
''')
rot = pose_dst[:3,:3]
print("\ndet(rot) =", np.linalg.det(rot))
print("(rot.T == rot^-1) =", np.allclose(rot.T, np.linalg.inv(rot)))
print("Hence, the rot comp. of the dst_pose HT is a rotation matrix.") 

