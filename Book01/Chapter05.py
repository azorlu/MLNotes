#%% [markdown]
# ##Logistic Regression

#%%
import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import h5py
import os
from PIL import Image
from scipy import ndimage

#%% [markdown]
# Load training and test datasets

#%%
def load_dataset():
    script_dir = os.getcwd()
    train_file_path = os.path.join(script_dir, 'MLNotes', 'datasets', 'train_catvnoncat.h5')
    test_file_path = os.path.join(script_dir, 'MLNotes', 'datasets', 'test_catvnoncat.h5')

    train_dataset = h5py.File(train_file_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_file_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#%%
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#%%
# display sample training image
def sample_training_data(index):
    c = classes[np.squeeze(train_set_y[:, index])].decode("utf-8")
    plt.imshow(train_set_x_orig[index])
    plt.text(2,10, c, fontsize=24)


#%%
sample_training_data(2)

#%%
sample_training_data(3)

#%%
# dataset info
def display_dataset_info():
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

#%%
display_dataset_info()

#%% [markdown]
# Reshape the training and test data sets so that 
# images of size (num_px, num_px, 3) are flattened into 
# single vectors of shape (num_px  ∗∗  num_px  ∗∗ 3, 1).
# How to  flatten a matrix X of shape (a,b,c,d) 
# to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a)
# X_flatten = X.reshape(X.shape[0], -1).T     
# where X.T is the transpose of X

#%%
def flatten_dataset():
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    return train_set_x_flatten, test_set_x_flatten

#%%
train_set_x_flatten, test_set_x_flatten = flatten_dataset()

#%%
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#%% [markdown]
# To represent color images, the red, green and blue channels 
# (RGB) must be specified for each pixel, and so the pixel value 
# is actually a vector of three numbers ranging from 0 to 255.
# One common preprocessing step in machine learning is to center and standardize 
# your dataset, meaning that you substract the mean of the whole numpy array 
# from each example, and then divide each example 
# by the standard deviation of the whole numpy array. 
# But for picture datasets, it is simpler and more convenient 
# and works almost as well to just divide 
# every row of the dataset by 255 (the maximum value of a pixel channel)

#%% [markdown]
# Standardized dataset

#%%
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#%% [markdown]
# Main steps to build a neural network
# 1. Define the model structure (such as number of input features)
# 2. Initialize the model's parameters
# 3. Loop:
# * Calculate current loss (forward propagation)
# * Calculate current gradient (backward propagation)
# * Update parameters (gradient descent)

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

#%% [markdown]
# Initialize parameters
# For image inputs, w will be of shape (num_px  ××  num_px  ××  3, 1)

#%%
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

#%% [markdown]
# Forward and Backward propagation
# Forward Propagation
# * Get X
# * Compute compute  A=σ(wTX+b)=(a(1),a(2),...,a(m−1),a(m))A=σ(wTX+b)=(a(1),a(2),...,a(m−1),a(m))
# * Calculate the cost function:  J=−1m∑mi=1y(i)log(a(i))+(1−y(i))log(1−a(i))J=−1m∑i=1my(i)log⁡(a(i))+(1−y(i))log⁡(1−a(i))

#%%
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    A = sigmoid(np.dot(w.T,X) + b)
    # compute cost
    cost = (-1/m) * np.sum( (Y *np.log(A)) + ((1-Y) * np.log(1-A)) )
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    # derivatives of w and b
    db = (1/m) * (np.sum(A-Y))
    dw = (1/m) * (np.dot(X, np.subtract(A,Y).T))
    
    assert(db.dtype == float)
    assert(dw.shape == w.shape)
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

#%% [markdown]
# Optimization
# The goal is to learn $w$ and $b$ by minimizing the cost function $J$. 
# For a parameter $\theta$, the update rule is 
# $ \theta = \theta - \alpha \text{ } d\theta$, 
# where $\alpha$ is the learning rate.

#%%
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

#%% [markdown]
# Prediction
# 1. Calculate  Ŷ =A=σ(wTX+b)Y^=A=σ(wTX+b)
# 2. Convert the entries of a into 0 (if activation <= 0.5) 
# or 1 (if activation > 0.5), 
# stores the predictions in a vector Y_prediction

#%%
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#%% [markdown]
# Model creation
# Merge all above functions into a model

#%%
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#%% [markdown]
# #Testing

#%%
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#%%
d["Y_prediction_test"]

#%%
# display sample test image and prediction result
def sample_test_data(index):
    num_px = train_set_x_orig.shape[1]
    c = classes[d["Y_prediction_test"][0,index]].decode("utf-8")
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    plt.text(2,10, c, fontsize=24)

#%%
sample_test_data(5)

#%%
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#%% [markdown]
# Choice of learning rate
# The learning rate  αα  determines how rapidly we update 
# the parameters. If the learning rate is too large 
# we may "overshoot" the optimal value. Similarly, if it is 
# too small we will need too many iterations to converge 
# to the best values. 
# That's why it is crucial to use a well-tuned learning rate.

#%%
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#%% [markdown]
# #Interpretation
# Different learning rates give different costs and thus different predictions results.
# If the learning rate is too large (0.01), the cost may oscillate up and down. 
# It may even diverge (though in this example, 
# using 0.01 still eventually ends up at a good value for the cost).
# A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. 
# It happens when the training accuracy is a lot higher than the test accuracy.
# In deep learning, we usually recommend that you:
# Choose the learning rate that better minimizes the cost function.
# If your model overfits, use other techniques to reduce overfitting