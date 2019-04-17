#%% [markdown]
# #Vectors and Matrices

#%%
import numpy as np 
import matplotlib.pyplot as plt 

#%% [markdown]
# Scalar is a 0 dimensional array

#%%
def np_array_info(x):
    print(x)
    print("type: ", type(x))
    print("dimension: ", np.ndim(x))
    print("shape: ", x.shape)

#%%
s = np.array(1)
np_array_info(s)

#%% [markdown]
# Vector is a 1 dimensional array

#%%
v = np.array([1, 0])
np_array_info(v)

#%% [markdown]
# Matrix is an N dimensional array

#%%
A = np.array([[1, 0], [0,1]])
np_array_info(A)

#%% [markdown]
# Create a 2 dimensional identity matrix

#%%
I = np.identity(2, int)
np_array_info(I)

#%% [markdown]
# Create matrix using eye function
#%%
E = np.eye(4, 5, k=-1, dtype=int)
np_array_info(E)

#%% [markdown]
# ndarray and matrix multiplications

#%%
A = np.array([0, 1, 2 ,1]).reshape(2,2)
B = np.array([-1, 2, 0 ,3]).reshape(2,2)
print("A")
print(A)
print("B")
print(B)

#%%
print("A * B gives elementwise (Hadamard) product")
print(A * B)
print("A @ B or np.dot(A, B) gives dot product")
print(A @ B)
print("Alternatively one can use np.mat(A) * np.mat(B)")
print(np.mat(A) * np.mat(B))
print("np.mat() converts a n dimensional array to a matrix")


#%% [markdown]
# Remember that matrix multiplication is not (in general) commutative
#%%
print("A @ B")
print(A @ B)
print("B @ A")
print(B @ A)

#%% [markdown]
# Dot product of vectors
#%%
v1 = np.array([1, 2])
v2 = np.array([1, 3])
print("v1")
print(v1)
print("v2")
print(v2)
print("v1 @ v2")
print(v1 @ v2)

#%% [markdown]
# Multiplication by broadcasting

#%%
A = np.arange(1, 10).reshape(3,3)
B = np.array([[1, 10, 100]])
print("A")
print(A)
print("B")
print(B)
print("A * B")
print(A * B)

#%% [markdown]
#Convert a row vector to column vector

#%%
v1 = np.array([1, 10, 100])
print("v1")
print(v1)
v2 = v1[:, np.newaxis]
print("v1[:, np.newaxis]")
print(v2)

#%% [markdown]
# Multiplication by row vector and a column vector

#%%
A = np.arange(1, 10).reshape(3,3)
v1 = np.array([1, 10, 100])
v2 = v1[:, np.newaxis]
print("\nA\n")
print(A)
print("\nv1\n")
print(v1)
print("\nA * v1 # multiply by row vector\n")
print(A * v1)
print("\nA * v1[:, np.newaxis] # multiply by column vector\n")
print(A * v2)


#%% [markdown]
# Flattening an ndarray
# see https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/

#%% [markdown]
# flatten returns copy of original array
#%%
A = np.arange(1, 10).reshape(3,3)
B = A.flatten()
print("A")
print(A)
print("B")
print(B)
B[[0,0]] = -1
print("First element of B is modified")
print(B)
print("A is not affected")
print(A)

#%% [markdown]
# ravel returns reference/view of original array
# ravel is faster, no extra memory is needed

#%%
A = np.arange(1, 10).reshape(3,3)
B = A.ravel()
print("A")
print(A)
print("B")
print(B)
B[[0,0]] = -1
print("First element of B is modified")
print(B)
print("A is affected")
print(A)

#%% [markdown]
# reshape returns a view
# https://www.numpy.org/devdocs/reference/generated/numpy.reshape.html


#%%
A = np.arange(1, 10).reshape(3,3)
B = A.reshape((-1,))
print("A")
print(A)
print("B")
print(B)
B[[0,0]] = -1
print("First element of B is modified")
print(B)
print("A is affected")
print(A)

#%% [markdown]
# concatenate: join a sequence of arrays along an existing axis.

#%%
M = np.arange(6).reshape(2,3)
N = np.arange(100, 106).reshape(2,3)
print("M")
print(M)
print("N")
print(N)
print("Concatenate M and N by axis=0")
print(np.concatenate((M, N)))
print("Concatenate M and N by axis=1")
print(np.concatenate((M, N), axis=1))


#%% [markdown]
# stack: join a sequence of arrays along __a new__ axis.

#%%
M = np.arange(6).reshape(2,3)
N = np.arange(100, 106).reshape(2,3)
print("M")
print(M)
print("N")
print(N)
print("Stack M and N by new axis=2")
print(np.stack((M, N), axis=2))

#%% [markdown]
# np.tile()

#%%
A = np.identity(2, int)
print("A")
print(A)
print("B = np.tile(A, (4,4))")
B = np.tile(A, (4,4))
print(B)

#%% [markdown]
# np.sum() and np.mean()

#%%
A = np.arange(1, 10).reshape(3,3)
print("A \n", A)
sum_by_column = np.sum(A, axis=0)
sum_by_row = np.sum(A, axis=1)
print("\n sum of A by column \n", sum_by_column)
print("\n sum of A by row \n", sum_by_row)
mean_by_column = np.mean(A, axis=0)
mean_by_row = np.mean(A, axis=1)
print("\n mean of A by column \n", mean_by_column)
print("\n mean of A by row \n", mean_by_row)

#%% [markdown]
# Use np.cov() to calculate covariance

#%%
A = np.array([[-1, 1, 3], [3, 1, -1]])
print("A \n", A)
cov = np.cov(A)
print("\n covariance matrix of A \n", cov)

#%% [markdown]
# #Distance and angle between two vectors

#%%
# Compute distance between two vectors x, y using the dot product
def distance(x,y):
    x = np.array(x, dtype=np.float).ravel()
    y = np.array(y, dtype=np.float).ravel()
    distance = ((x - y).T @ (x - y)) ** 0.5
    return distance

#%%
# Compute the angle between two vectors x, y using the dot product
def angle(x, y):
    angle = np.arccos((x.T @ y) / ((x.T @ x) * (y.T @ y)) ** 0.5)
    return angle

#%% 
# sanity check
a = np.array([1,0])
b = np.array([0,1])
np.testing.assert_almost_equal(distance(a, b), np.sqrt(2))
assert((angle(a,b) / (np.pi * 2) * 360.) == 90)
print("correct")






