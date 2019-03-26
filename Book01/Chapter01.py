#%% [markdown]
# #NumPy and The Maths

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
print("A * B gives elementwise product")
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
print("A")
print(A)
print("A * v1 # multiply by row vector")
print(A * v1)
print("A * v1[:, np.newaxis] # multiply by column vector")
print(A * v2)
