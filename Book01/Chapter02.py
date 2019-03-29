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

#%%
# np.random.randint() method return sample(s) from a discrete uniform distribution
#%%
# naive approach to simulate rolling of a fair die
R = np.random.randint(1, 7, (3, 10))
R

#%%
# np.random.randn() method return sample(s) from standard normal distribution
# which is a univariate “normal” (Gaussian) distribution of mean 0 and variance 1
# same as np.random.standard_normal
#%%
R = np.random.randn(3,3)
R

#%%
# For random samples from N(\mu, \sigma^2) use
# sigma * np.random.randn(...) + mu

#%%
sigma = 2,
mu = 175
R = sigma * np.random.randn(3,3) + mu
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