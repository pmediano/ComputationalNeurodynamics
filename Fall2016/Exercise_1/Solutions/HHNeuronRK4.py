"""
Computational Neurodynamics
Exercise 1

Simulates the Hodgkin-Huxley neuron model using the RK4 method,
assuming a resting potential of 0mV, as in the 1952 paper. To get a
realistic resting potential of -65mV, refer to HHNeuronDemo.py

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

## Create time points
dt   = 0.01
Tmin = 0
Tmax = 100
T    = np.arange(Tmin, Tmax+dt, dt)

## Parameters of the Hodgkin-Huxley model
gNa = 120.0
gK  = 36.0
gL  = 0.3
ENa = 115.0
EK  = -12.0
EL  = 10.6
C   = 1.0

# Base current
I = 10.0

# Make a state vector that has a (v, m, n, h) pair for each timestep
s = np.zeros((len(T), 4))

## Initial Values
s[0] = np.array([-10.0, 0., 0., 0.])


def s_dt(s1, I):
  v, m, n, h = s1

  alpham = (2.5 - 0.1*v)/(np.exp(2.5 - 0.1*v) - 1)
  alphan = (0.1 - 0.01*v)/(np.exp(1.0 - 0.1*v) - 1)
  alphah = 0.07*np.exp(-v/20.0)

  betam = 4.0*np.exp(-v/18.0)
  betan = 0.125*np.exp(-v/80.0)
  betah = 1.0/(np.exp(3.0-0.1*v) + 1.0)

  Ik = gNa*(m**3)*h*(v-ENa) + gK*(n**4)*(v-EK) + gL*(v-EL)

  v_dt = (I - Ik)/C
  m_dt = alpham*(1-m) - betam*m
  n_dt = alphan*(1-n) - betan*n
  h_dt = alphah*(1-h) - betah*h

  res = np.array([v_dt, m_dt, n_dt, h_dt])
  return res


## SIMULATE
for t in xrange(len(T)-1):

  # Calculate the four constants of Runge-Kutta method
  k_1 = s_dt(s[t], I)
  k_2 = s_dt(s[t] + 0.5*dt*k_1, I)
  k_3 = s_dt(s[t] + 0.5*dt*k_2, I)
  k_4 = s_dt(s[t] + dt*k_3, I)

  s[t+1] = s[t] + (1.0/6)*dt*(k_1 + 2*k_2 + 2*k_3 + k_4)


v = s[:, 0]

## Plot the membrane potential
plt.plot(T, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Hodgkin-Huxley Neuron')
plt.show()

