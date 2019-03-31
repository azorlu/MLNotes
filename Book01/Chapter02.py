#%% [markdown]
# #NumPy and Random numbers
# Note: __random__ module should be used for simulation only.
# Use __secrets__ module to get cryptographically strong pseudo random numbers  
# np.random module is fast but deterministic (Mersenne twister sequence?)

#%%
import random
import numpy as np
import matplotlib.pyplot as plt

#%% [markdown]
# np.random.random() is an alias to np.random.random_sample()
# np.random.random() method return sample(s) from a continuous uniform distribution
#%%
R = np.random.random((3,3))
R

#%% [markdown]
# np.random.randint() method return sample(s) from a discrete uniform distribution
#%%
# naive approach to simulate rolling of a fair die
R = np.random.randint(1, 7, (3, 10))
R

#%% [markdown]
# np.random.randn() method return sample(s) from standard normal distribution
# which is a univariate “normal” (Gaussian) distribution of mean 0 and variance 1
# same as np.random.standard_normal
#%%
R = np.random.randn(3,3)
R

#%% [markdown]
# For random samples from N(\mu, \sigma^2) use
# sigma * np.random.randn(...) + mu

#%%
sigma = 1,
mu = 0.5
R = sigma * np.random.randn(3,3) + mu
R

#%% [markdown]
# np.random.choice() method generates a random sample from a given 1-D array

#%%
# uniform random sample without replacement
R = np.random.choice(5, 3, replace=False)
R

#%% [markdown]
# non-uniform random sample without replacement
# probabilities should sum to one

#%%
R = np.random.choice(5, 3, replace=False, p=[0.2, 0.3, 0.3, 0.1, 0.1])
R

#%% 
# Using Dirichlet distribution
R = np.random.choice(5, 3, replace=False, p=np.random.dirichlet(np.ones(5)))
R

#%% [markdown]
# Generate random numbers satisfying sum-to-one condition
#%% 
# Normalize a list of reandom values
R = np.random.random(10)
R /= R.sum()
R

#%% [markdown]
# Use np.random.dirichlet to generate random numbers satisfying sum-to-one condition
# A Dirichlet-distributed random variable can be seen as a multivariate generalization of a Beta distribution.

#%% 
R = np.random.dirichlet(np.ones(10))
R

#%% [markdown]
# https://en.wikipedia.org/wiki/Dirichlet_distribution#String_cutting
# cut strings (each of initial length 1.0) into K pieces with different lengths, 
# where each piece had a designated average length, 
# but allowing some variation in the relative sizes of the pieces

#%%
num = 10
R = np.random.dirichlet((10, 5, 3), num).transpose()
plt.barh(range(num), R[0])
plt.barh(range(num), R[1], left=R[0], color='g')
plt.barh(range(num), R[2], left=R[0]+R[1], color='r')

#%% [markdown]
# Use np.random.normal() to draw random samples from a normal (Gaussian) distribution.
# https://en.wikipedia.org/wiki/Normal_distribution
# The probability density for the Gaussian distribution is
# p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }} e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },
# where \mu is the mean and \sigma the standard deviation. The square of the standard deviation, \sigma^2, is called the variance.

#%%
mu, sigma = 0, 0.1
R = np.random.normal(mu, sigma, 10)
R

#%% [markdown]
# Histogram of the samples, along with the probability density function

#%%
mu, sigma = 0, 0.1
R = np.random.normal(mu, sigma, 10**3)
count, bins, ignored = plt.hist(R, 50, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    linewidth=2, color='r')

#%% [markdown]
# np.random.standard_normal method return sample(s) from standard normal distribution
# which is a univariate “normal” (Gaussian) distribution of mean 0 and variance 1
# same as np.random.randn()
#%%
R = np.random.standard_normal(5)
R

#%%
def random_points_in_circle(
    radius=1, center_x=0, center_y=0, number_of_points=10**3, is_half_circle=False):
    n = 1 if is_half_circle else 2    
    rho = radius * np.sqrt(np.random.uniform(0, 1, number_of_points))
    theta = np.random.uniform(0, n * np.pi, number_of_points)
    xs = center_x + rho * np.cos(theta)
    ys = center_y + rho * np.sin(theta)
    return (xs, ys)

xs, ys = random_points_in_circle()
plt.scatter(xs, ys, s=1)
plt.show()