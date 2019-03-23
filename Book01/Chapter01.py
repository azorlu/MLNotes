#%% [markdown]
# #NumPy and The Maths

#%%
import numpy as np 
import matplotlib.pyplot as plt 

#%% [markdown]
# ndarray and matrix multiplications

#%%
A = np.array([0, 1, 2 ,1]).reshape(2,2)
print("A")
A

#%%
B = np.array([-1, 2, 0 ,3]).reshape(2,2)
print("B")
B

#%%
print("A * B gives elementwise product")
A * B

#%%
print("A @ B gives matrix product")
A @ B
