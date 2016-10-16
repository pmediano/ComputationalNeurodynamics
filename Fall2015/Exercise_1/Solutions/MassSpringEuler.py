"""
Computational Neurodynamics
Exercise 1

Solution of the damped mass spring oscillator
system for Question 1(b)

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

# List of parameters
m  = 1
c  = 0.1
k  = 1
dt = 0.01

# Create time points
Tmin = 0
Tmax = 100
T    = np.arange(Tmin, Tmax+dt, dt)
y    = np.zeros(len(T))
dy   = np.zeros(len(T))
dy2  = np.zeros(len(T))

# Initial conditions
y[0]   = 1
dy[0]  = 0
dy2[0] = (-c/m)*dy[0] - (k/m)*y[0]

# Euler method
for t in xrange(1, len(T)):
  dy2[t] = (-c/m)*dy[t-1] - (k/m)*y[t-1]
  dy[t]  = dy[t-1] + dt*dy2[t-1]
  y[t]   = y[t-1] + dt*dy[t-1]

# Plot the results
plt.plot(T, y, 'r')
plt.xlabel('t')
plt.ylabel('y')
plt.show()

