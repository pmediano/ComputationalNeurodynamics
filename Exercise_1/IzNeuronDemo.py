"""
Computational Neurodynamics
Exercise 1

Simulates Izhikevich's neuron model using the Euler method.
Parameters for regular spiking, fast spiking and bursting
neurons extracted from:

http://www.izhikevich.org/publications/spikes.htm

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

# Create time points
Tmin = 0
Tmax = 200   # Simulation time
dt   = 0.01  # Step size
T    = np.arange(Tmin, Tmax+dt, dt)

# Base current
I = 10

## Parameters of Izhikevich's model (regular spiking)
a = 0.02
b = 0.2
c = -65
d = 8

## Parameters of Izhikevich's model (fast spiking)
# a = 0.02
# b = 0.25
# c = -65
# d = 2

## Parameters of Izhikevich's model (bursting)
# a = 0.02
# b = 0.2
# c = -50
# d = 2

v = np.zeros(len(T))
u = np.zeros(len(T))

## Initial values
v[0] = -65
u[0] = -1

## SIMULATE
for t in xrange(len(T)-1):
  # Update v and u according to Izhikevich's equations
  v[t+1] = v[t] + dt*(0.04*v[t]**2 + 5*v[t] + 140 - u[t] + I)
  u[t+1] = u[t] + dt*(a * (b*v[t] - u[t]))

  # Reset the neuron if it has spiked
  if v[t+1] >= 30:
    v[t]   = 30          # Add a Dirac pulse for visualisation
    v[t+1] = c           # Reset to resting potential
    u[t+1] = u[t+1] + d  # Update recovery variable


## Plot the membrane potential
plt.subplot(211)
plt.plot(T, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (mV)')
plt.title('Izhikevich Neuron')

# Plot the reset variable
plt.subplot(212)
plt.plot(T, u)
plt.xlabel('Time (ms)')
plt.ylabel('Reset variable u')
plt.show()

