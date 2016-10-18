"""
Computational Neurodynamics
Exercise 2

Simulates the movement of a robot with differential wheels under the
control of a spiking neural network. The simulation runs for a very
long time --- if you get bored, press Ctrl+C a couple of times.

**Note**: this code may not work properly in Ubuntu 16.04. See RobotRun4L-u16.py
for a 16.04-compatible version.

(C) Murray Shanahan et al, 2016
"""

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from Environment import Environment
from RobotConnect4L import RobotConnect4L
from RobotUpdate import RobotUpdate


## Create the environment
print 'Initialising environment'
xmax = 100
ymax = 100
Env = Environment(15, 10, 20, xmax, ymax)

## Robot controller
print 'Initialising robot controller'
Ns = 4  # Sensor neurons. Try 1, 4, and 8
Nm = 4  # Motor neurons. Try 1, 4, and 8
N = 2*Ns + 2*Nm
net  = RobotConnect4L(Ns, Nm)

Dmax = 5      # Maximum synaptic delay
Ib   = 30     # Base current
Rmax = 40     # Estimated peak motor firing rate in Hz
Umin = 0.025  # Minimum wheel velocity in cm/ms
Umax = Umin + Umin/6.0  # Maximum wheel velocity

# Simulation parameters
Tmax = 20000  # Simulation time in milliseconds
dt   = 100    # Robot step size in milliseconds

# Initialise record of membrane potentials
v = np.zeros((dt, N))

# Initialise record of robot positions
T = np.arange(0, Tmax, dt)
x = np.zeros(len(T)+1)
y = np.zeros(len(T)+1)
w = np.zeros(len(T)+1)
w[0] = np.pi/4

# Size of layers
N1 = Ns
N2 = Ns
N3 = Nm
N4 = Nm

print 'Preparing simulation'

plt.figure(1)

# Draw Environment
plt.figure(2)
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.title('Robot controlled by spiking neurons')
plt.xlabel('X')
plt.ylabel('Y')
for Ob in Env.Obs:
  plt.scatter(Ob['x'], Ob['y'], s=np.pi*(Ob['r']**2), c='lime')


plt.ion()
plt.show()


## SIMULATE
print 'Start simulation'
for t in xrange(len(T)):
  # Input from Sensors
  SL, SR = Env.GetSensors(x[t], y[t], w[t])

  RL_spikes = 0.0
  RR_spikes = 0.0
  for t2 in xrange(dt):

    # Deliver stimulus as a Poisson spike stream to the sensor neurons and
    # noisy base current to the motor neurons
    I = np.hstack([rn.poisson(SL*15, N1), rn.poisson(SR*15, N2),
                   5*rn.randn(N3), 5*rn.randn(N4)])

    # Update network
    net.setCurrent(I)
    fired = net.update()

    RL_spikes += np.sum(np.logical_and(fired > (N1+N2), fired < N1+N2+N3))
    RR_spikes += np.sum(fired > (N1+N2+N3))

    # Maintain record of membrane potential
    v[t2,:],_ = net.getState()

  # Output to motors
  # Calculate motor firing rates in Hz
  RL = 1.0*RL_spikes/(dt*N3)*1000.0
  RR = 1.0*RR_spikes/(dt*N4)*1000.0

  # Set wheel velocities (as fractions of Umax)
  UL = (Umin/Umax + RL/Rmax*(1 - Umin/Umax))
  UR = (Umin/Umax + RR/Rmax*(1 - Umin/Umax))

  # Update Environment
  x[t+1], y[t+1], w[t+1] = RobotUpdate(x[t], y[t], w[t], UL, UR,
                                       Umax, dt, xmax, ymax)

  ## PLOTTING
  # Plot membrane potential
  plt.figure(1)
  plt.clf()

  plt.subplot(221)
  plt.plot(v[:,0:N1])
  plt.subplot(221)
  plt.title('Left sensory neurons')
  plt.ylabel('Membrane potential (mV)')
  plt.ylim(-90, 40)

  plt.subplot(222)
  plt.plot(v[:,(N1+1):(N1+N2)])
  plt.title('Right sensory neurons')
  plt.ylim(-90, 40)

  plt.subplot(223)
  plt.plot(v[:,(N1+N2+1):(N-N4)])
  plt.title('Left motor neurons')
  plt.ylabel('Membrane potential (mV)')
  plt.ylim(-90, 40)
  plt.xlabel('Time (ms)')

  plt.subplot(224)
  plt.plot(v[:,(N-N4+1):N])
  plt.title('Right motor neurons')
  plt.ylim(-90, 40)
  plt.xlabel('Time (ms)')

  plt.draw()

  # Plot robot trajectory
  plt.figure(2)
  plt.scatter(x, y, marker='.')
  plt.draw()

