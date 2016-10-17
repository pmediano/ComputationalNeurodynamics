"""
Computational Neurodynamics
Exercise 1

Simulates the Hodgkin-Huxley neuron model using the Euler method,
assuming a resting potential of 0mV, as in the 1952 paper. To get a
realistic resting potential of -65mV, use the code in comments.

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

## Create time points
dt   = 0.01
Tmin = 0
Tmax = 100
T = np.arange(Tmin, Tmax+dt, dt)
v = np.zeros(len(T))

## Parameters of the Hodgkin-Huxley model
gNa = 120.0
gK  = 36.0
gL  = 0.3
ENa = 115.0
EK  = -12.0
EL  = 10.6
C   = 1.0
# ENa = 115 - 65
# EK  = -12 - 65
# EL  = 10.6 - 65

# Base current
I = 10.0

## Initial values
v[0] = -10.0
# v[0] = -75.0
m = 0.0
n = 0.0
h = 0.0

## Simulate
for t in xrange(len(T)-1):
  # Update v according to Hodgkin-Huxley equations

  alphan = (0.1 - 0.01*v[t])/(np.exp(1.0 - 0.1*v[t])-1)
  alpham = (2.5 - 0.1*v[t])/(np.exp(2.5 - 0.1*v[t])-1)
  alphah = 0.07*np.exp(-v[t]/20.0)

  betan = 0.125*np.exp(-v[t]/80.0)
  betam = 4.0*np.exp(-v[t]/18.0)
  betah = 1.0/(np.exp(3.0-0.1*v[t])+1.0)

#   alphan = (0.1-0.01*(v[t]+65))/(np.exp(1-0.1*(v[t]+65))-1)
#   alpham = (2.5-0.1*(v[t]+65))/(np.exp(2.5-0.1*(v[t]+65))-1)
#   alphah = 0.07*np.exp(-(v[t]+65)/20)
#
#   betan = 0.125*np.exp(-(v[t]+65)/80)
#   betam = 4*np.exp(-(v[t]+65)/18)
#   betah = 1/(np.exp(3-0.1*(v[t]+65))+1)

  m = m + dt*(alpham*(1-m) - betam*m)
  n = n + dt*(alphan*(1-n) - betan*n)
  h = h + dt*(alphah*(1-h) - betah*h)

  Ik = gNa*(m**3)*h*(v[t]-ENa) + gK*(n**4)*(v[t]-EK) + gL*(v[t]-EL)

  v[t+1] = v[t] + dt*(-Ik+I)/C


## Plot the membrane potential
plt.plot(T, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Hodgkin-Huxley Neuron')
plt.show()

