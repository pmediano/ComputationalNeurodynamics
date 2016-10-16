"""
Computational Neurodynamics
Exercise 2

Simulates a Quadratic Integrate-and-Fire neuron using
the Euler method.

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt


## Create time points
Tmin = 0
Tmax = 50   # Simulation time
dt   = 0.2  # Step size
T    = np.arange(Tmin, Tmax+dt, dt)

# Base current
I = 20

## Parameters of QIF model (regular spiking)
R   = 1.0
tau = 5
vr  = -65
vc  = -50
a   = 0.2

v = np.zeros(len(T))

## Initial values
v[0] = -65

## SIMULATE
for t in range(len(T)-1):
  # Update v according to QIF equations
  v[t+1] = v[t] + dt*((a*(vr-v[t])*(vc-v[t]) + R*I)/tau)

  # Reset the neuron if it has spiked
  if v[t+1] >= 30:
    v[t]   = 30  # Add Dirac pulse for visualisation
    v[t+1] = vr


## Plot the membrane potential
plt.plot(T, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (mV)')
plt.title('Quadratic Integrate-and-Fire Neuron')
plt.show()

