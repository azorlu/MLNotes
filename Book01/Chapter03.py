#%% [markdown]
# #NumPy, SciPy and Linear Algebra

#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import linalg

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
# Rank is the number of linearly independent directions
def print_rank(A):
    print("\n Matrix A:\n", A)
    print("Rank of A:", np.linalg.matrix_rank(A))

#%%
print_rank(np.array([[0,0], [0,0]]))
print_rank(np.array([[1,0], [1,0]]))
print_rank(np.array([[1,0], [1,1]]))

#%%
# Calculate eigenvectors and eigenvalues
w, v = np.linalg.eig(A)
print("\nEigenvalues of A:\n", w)
print("\nEigenvectors of A:\n", v)

#%%
# Calculate vector norms
a = np.array([2,1,-2])
print("\nvector a:\n", a)
print("\nTaxicab (Manhattan) norm of vector a:\n", 
    np.linalg.norm(a, ord=1))
print("\nFrobenius (Euclidean) norm (magnitude) of vector a:\n", 
    np.linalg.norm(a))
print("\nMax norm of vector a:\n", 
    np.linalg.norm(a, ord=np.inf))

#%%
# Calculate matrix norms
A = np.array([[1,0,0], [2,2,-8], [0, -2, 2]])
print("\nmatrix A:\n", A)
print("\nTaxicab (Manhattan) norm of matrix A:\n", 
    np.linalg.norm(A, ord=1))
print("\nFrobenius (Euclidean) norm (magnitude) of matrix A:\n", 
    np.linalg.norm(A))
print("\nMax norm of matrix A:\n", 
    np.linalg.norm(A, ord=np.inf))

#%%
# Get upper and lower triangular matrices and diaognal matrix
A = np.arange(1,10).reshape((3,3))
print("\nmatrix A:\n", A)
print("\nUpper triangular matrix of A:\n", np.triu(A))
print("\nLower triangular matrix of A:\n", np.tril(A))
print("\nDiagonal matrix created from diagonal vector of A:\n", np.diag(np.diag(A)))

#%% [markdown]
# Orthogonal matrices
# An orthogonal matrix is a square matrix 
# whose columns and rows are orthogonal unit vectors
# https://en.wikipedia.org/wiki/Orthogonal_matrix
# A matrix Q is orthogonal if its transpose is equal to its inverse

#%%
Q = np.array([[0, -1], [1, 0]])
print("\nmatrix Q:\n", Q)
print("\nTranspose of matrix Q:\n", Q.T)
print("\nInverse of matrix Q:\n", np.linalg.inv(Q))
print("\nDot product of Q and its transpose:\n", Q.dot(Q.T))

#%% 
# 2x2 rotation matrix is an orthogonal matrix
def rot2D(theta):
    c = np.cos(np.radians(theta))
    s = np.sin(np.radians(theta))
    return np.array([[c, -s], [s, c]])

#%% 
# 90 degrees rotational matrix
R = rot2D(90)
print("\nmatrix R:\n", R)
print("\nDot product of R and its transpose:\n", R.dot(R.T))

#%% [markdown]
# Calculate Fibonacci numbers using matrices
# by taking the power of the Fibonacci matrix

#%%
# n should be less than 1476
def fib(n):
    F = np.array([[1, 1], [1, 0]], dtype=float)
    return np.linalg.matrix_power(F, n)[0,0]

#%%
n = 1475
print(str(n) + "th Fibonacci number:\n", fib(n))

#%% [markdown]
# __Sparse Matrices__

#%% [markdown]
# Compressed Sparse Row: 
# The sparse matrix is represented using three one-dimensional arrays 
# for the non-zero values, the extents of the rows, and the column indexes.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix

#%%
n = 7
S = np.zeros((n,n))
for _ in range(n):
    S[_, np.random.randint(0, n)] = 1
print("\nmatrix S:\n", S)
S2 = sparse.csr_matrix(S)
print("\nCSR matrix S2:\n", S2)

#%%
def matrix_sparsity(A):
    return 1.0 - (np.count_nonzero(A) / A.size)

print("\nSparsity of S:\n", matrix_sparsity(S))

#%%
# Generate a sparse matrix of the given shape and density with randomly distributed values.
A = sparse.random(5, 4, density=0.15)
print("\nrandom sparse matrix A:\n", A)
print("\nA reconstructed:\n", A.todense())

#%% [markdown]
# __Tensors__
# https://en.wikipedia.org/wiki/Tensor
# A vector is a first order tensor
# A matrix is a second order tensor

#%%
T1 = np.array([1,2])
T2 = np.array([-1,3])
print("\nTensor T1:\n", T1)
print("\nAnother Tensor T2:\n", T2)
print("\nTensor product of T1 and T2 at axes=0:\n", 
    np.tensordot(T1, T2, axes=0))
print("\nTensor dot product of T1 and T2 at axes=1:\n", 
    np.tensordot(T1, T2, axes=1))


