#%% [markdown]
# ##K Nearest Neighbors

#%%
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

#%% [markdown]
# Load Iris datasets
# https://archive.ics.uci.edu/ml/datasets/iris
# 4 attributes (features): feature vector x ∈ ℝ4 
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm
# target y∈ℤ is the class of the flower (3 classes)
# Iris Setosa 
# Iris Versicolour 
# Iris Virginica


#%%
iris = datasets.load_iris()

#%%
print('data shape is {}'.format(iris.data.shape))
print('class shape is {}'.format(iris.target.shape))

#%%
# use first two version for simplicity
X = iris.data[:, :2] 
y = iris.target
# scatter plot of the iris dataset 
# The x and y axis represent the sepal length 
# and sepal width of the dataset, 
# and the color of the points represent 
# the different classes of flowers.
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000',  '#00FF00', '#0000FF'])
K = 3
x = X[-1]
fig, ax = plt.subplots(figsize=(4,4))
for i, iris_class in enumerate(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']):
    idx = y==i
    ax.scatter(X[idx,0], X[idx,1], 
               c=cmap_bold.colors[i], edgecolor='k', 
               s=20, label=iris_class)
ax.set(xlabel='sepal length (cm)', ylabel='sepal width (cm)')
ax.legend()

#%% [markdown]
# KNN: Given a training set  X ∈ ℝ N×D  and  y ∈ ℤ N , 
# predict the label of a new point  x ∈ ℝ D  
# as the label of the majority of its "K nearest neighbor" 
# by some distance measure (e.g the Euclidean distance). 
# N  is the number of data points in the dataset, 
# and  D  is the dimensionality of the data.

#%%
# Compute distance between two vectors x, y using the dot product
def distance(x,y):
    x = np.array(x, dtype=np.float).ravel()
    y = np.array(y, dtype=np.float).ravel()
    distance = ((x - y).T @ (x - y)) ** 0.5
    return distance

def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)
    
    Returns
    --------
    distance_matrix: matrix of shape (N, M), each entry distance_matrix[i,j] is the distance between
    ith row of X and the jth row of Y (we use the dot product to compute the distance).
    """
    N, D = X.shape
    M, _ = Y.shape
    distance_matrix = np.zeros((N, M), dtype=np.float)
    # ToDo: vectorize computation, do not use for loops
    for i in range(0, N):
        for j in range(0, M):
            distance_matrix[i, j] = distance(X[i, :], Y[j, :])
    return distance_matrix

#%%
def KNN(k, X, y, x):
    """K nearest neighbors
    k: number of nearest neighbors
    X: training input locations
    y: training labels
    x: test input
    """
    N, D = X.shape
    M, _ = x.shape
    num_classes = len(np.unique(y))
    # compute the pairwise distance matrix
    dist = np.zeros((N, M))

    # Find indices for the k closest flowers
    idx = np.argsort(dist.T, axis=1)[:, :K]

    # Next we make the predictions
    # Vote for the major class
    ypred = np.zeros((M, num_classes))

    # find the labels of the k nearest neighbors
    # classes = y[np.argsort(dist)][:k] 
    # for c in np.unique(classes):
        # compute the correct prediction
        # ypred[c] = 0

    for m in range(M):
        classes = y[idx[m]]    
        for k in np.unique(classes):
            ypred[m, k] = len(classes[classes == k]) / K 
    
    return np.argmax(ypred, axis=1)

#%%
# plots the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

ypred = []
for data in np.array([xx.ravel(), yy.ravel()]).T:
    ypred.append(KNN(K, X, y, data.reshape(1,2)))

fig, ax = plt.subplots(figsize=(4,4))

ax.pcolormesh(xx, yy, np.array(ypred).reshape(xx.shape), cmap=cmap_light)
ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)