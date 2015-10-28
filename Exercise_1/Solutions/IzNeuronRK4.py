"""
Computational Neurodynamics
Exercise 1

Simulates Izhikevich's neuron model using the Runge-Kutta 4 method.
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

## Make a state vector that has a (v, u) pair for each timestep
s = np.zeros((len(T), 2))

## Initial values
s[0, 0] = -65
s[0, 1] = -1


# Note that s1[0] is v, s1[1] is u. This is Izhikevich equation in vector form
def s_dt(s1, I):
  v_dt = 0.04*(s1[0]**2) + 5*s1[0] + 140 - s1[1] + I
  u_dt = a*(b*s1[0] - s1[1])
  return np.array([v_dt, u_dt])


## SIMULATE
for t in range(len(T)-1):

  # Calculate the four constants of Runge-Kutta method
  k_1 = s_dt(s[t], I)
  k_2 = s_dt(s[t] + 0.5*dt*k_1, I)
  k_3 = s_dt(s[t] + 0.5*dt*k_2, I)
  k_4 = s_dt(s[t] + dt*k_3, I)

  s[t+1] = s[t] + (1.0/6)*dt*(k_1 + 2*k_2 + 2*k_3 + k_4)

  # Reset the neuron if it has spiked
  if s[t+1, 0] >= 30:
    s[t, 0]   = 30  # Add a Dirac pulse for visualisation
    s[t+1, 0] = c   # Reset to resting potential
    s[t+1, 1] += d  # Update recovery variable


v = s[:, 0]
u = s[:, 1]

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

