# This script demonstrates the use of discrete least squares approximation on a small dataset of ${x_i,y_i}$ values
# I use the built in numpy.linalg.solve function to solve the resulting normal equations, resulting in $a, b$
# which determines the linear model $Y = ax + b$

import numpy as np

dataset1 = np.array([[1.15, 1.10], [0.39, 0.41], [3.59, 3.36], [2.10, 2.03], [1.68, 1.80], [4.29, 4.11], [3.85, 3.50]])
n = len(dataset1)

# Calculate coefficients for normal equations based on dataset
u = 0
v = 0
w = 0
z = 0
for i in range(n):
  u = u + dataset1[i, 0] # u = sum(x_i)
  v = v + dataset1[i, 1] # v = sum(y_i)
  w = w + dataset1[i, 0]**2 # w = sum(x_i^2)
  z = z + dataset1[i, 0]*dataset1[i, 1] # z = sum(x_i * y_i)

# Set up normal equations in matrix form
A = np.array([[n, u], [u, w]])
b = np.array([[v, z]]).transpose()

x = np.linalg.solve(A,b)

# Sampled output for the resulting linear model over $[0,1]$
Y = np.zeros(20)
for t in range(20):
  Y[t] = x[1]*0.05*t + x[0]
print(Y)
