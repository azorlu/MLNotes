#%% [markdown]
# #NumPy and Random numbers

#%%
import numpy as np
import matplotlib.pyplot as plt

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