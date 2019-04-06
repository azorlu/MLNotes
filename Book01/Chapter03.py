#%% [markdown]
# #NumPy Linear Algebra

#%%
import numpy as np 
import matplotlib.pyplot as plt 

#%% [markdown]
# Linear algebra solves linear equations
# y = A * b
# where y is output variable (vector), 
# A is dataset (matrix) and b is model coeffecients vector

#%%
# Some basic vector and matrix functions
A = np.array([[1, 0, 2], [1, 2, 1], [0, 3, 1]])
print("A: \n", A)
print("Rank of A:", np.linalg.matrix_rank(A))
print("\nTrace of A:", np.trace(A))
print("\nDeterminant of A:", np.linalg.det(A))
print("\nInverse of A:\n", np.linalg.inv(A))
print("\nMatrix A raised to power 2:\n",
    np.linalg.matrix_power(A, 2))

#%%
# Calculate eigenvectors and eigenvalues
w, v = np.linalg.eig(A)
print("\nEigenvalues of A:\n", w)
print("\nEigenvectors of A:\n", v)




