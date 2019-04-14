#%% [markdown]
# ##Machine Learning

#%%
import math
import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy import linalg

#%% [markdown]
# Sigmoid function
# σ(x)= 1 / (1 + exp(-x)) 

#%%
# use np.exp instead of math.exp
# so that the function works with vectors as well as scalars
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#%%
xs = np.linspace(-5,5)
plt.plot(xs, sigmoid(xs))
plt.show()

#%% [markdown]
# Sigmoid derivative function
# Derivative of the sigmoid function 
# with respect to its input x
# σ′(x) = σ(x)(1−σ(x)) 

#%%
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1-s)

#%%
xs = np.linspace(-5,5)
plt.plot(xs, sigmoid_derivative(xs))
plt.show()

#%% [markdown]
# Unroll 3D array (image) into a 1D vector

#%%
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v

#%% [markdown]
# Normalize vectors

#%%
def normalizeRows(x):
    """
    A function that normalizes each row of the matrix x
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    # Divide x by its norm.
    x = x/x_norm
    return x

#%% [markdown]
# Softmax function

#%%
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    # Apply exp() element-wise to x.
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum   
    return s

#%% [markdown]
# L1 and L2 loss functions
# L1(ŷ ,y) = ∑i=0 m |y(i)−ŷ (i)|
# L2(ŷ ,y) = ∑i=0 m (y(i)−ŷ (i))**2

#%%
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    loss = np.sum(abs(y-yhat))
    return loss

#%%
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    loss = np.sum(np.power((y-yhat),2))
    return loss