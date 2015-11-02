"""
Computational Neurodynamics
Exercise 2

Simulates the movement of a robot with differential wheels under the
control of a spiking neural network. The simulation runs for a very
long time --- if you get bored, press Ctrl+C a couple of times.

(C) Murray Shanahan et al, 2015
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
print 'Initialising Robot Controller'
Ns = 4  # Sensor neurons. Try 1, 4, and 8
Nm = 4  # Motor neurons. Try 1, 4, and 8
net  = RobotConnect4L(Ns, Nm)

Dmax = 5      # Maximum synaptic delay
Ib   = 30     # Base current
Rmax = 40     # Estimated peak motor firing rate in Hz
Umin = 0.025  # Minimum wheel velocity in cm/ms
Umax = Umin + Umin/6.0  # Maximum wheel velocity

## Initialise layers
for lr in xrange(net.Nlayers):
  net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
  net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
  net.layer[lr].firings = np.array([])

# Simulation parameters
Tmax = 20000  # Simulation time in milliseconds
dt   = 100    # Robot step size in milliseconds

# Initialise record of membrane potentials
v = {}
for lr in xrange(net.Nlayers):
  v[lr] = np.zeros([dt, net.layer[lr].N])


# Initialise record of robot positions
T = np.arange(0, Tmax, dt)
x = np.zeros(len(T)+1)
y = np.zeros(len(T)+1)
w = np.zeros(len(T)+1)
w[0] = np.pi/4

# Size of layers
N0 = net.layer[0].N
N1 = net.layer[1].N
N2 = net.layer[2].N
N3 = net.layer[3].N
L  = net.Nlayers

print 'Preparing Simulation'

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
print 'Start Simulation'
for t in xrange(len(T)):
  # Input from Sensors
  # SL, SR = RobotGetSensors(Env, x[t], y[t], w[t], xmax, ymax)
  SL, SR = Env.GetSensors(x[t], y[t], w[t])

  # Carry over firings that might not have reached their targets yet
  for lr in xrange(L):
    firings = []
    for f in net.layer[lr].firings:
      # discard all earlier firings
      if f[0] > dt-Dmax:
        # also decrease the time so that it is in -Dmax to -1
        f[0] = f[0] - dt
        firings.append(f)
    net.layer[lr].firings = np.array(firings)

  for t2 in xrange(dt):
    # Deliver stimulus as a Poisson spike stream
    net.layer[0].I = rn.poisson(SL*15, N0)
    net.layer[1].I = rn.poisson(SR*15, N1)

    # Deliver noisy base current
    net.layer[2].I = 5*rn.randn(N2)
    net.layer[3].I = 5*rn.randn(N3)

    # Update network
    net.Update(t2)

    # Maintain record of membrane potential
    for lr in xrange(L):
      v[lr][t2, :] = net.layer[lr].v

  # Discard carried over firings with time less than 0
  for lr in xrange(L):
    firings = []
    for f in net.layer[lr].firings:
      if f[0] > 0:
        firings.append(f)
    net.layer[lr].firings = np.array(firings)

  # Add Dirac pluses (mainly for presentation)
  for lr in xrange(L):
    firings = net.layer[lr].firings
    if firings.size != 0:
      v[lr][firings[:, 0], firings[:, 1]] = 30

  # Output to motors
  # Calculate motor firing rates in Hz
  RL = 1.0*len(net.layer[2].firings)/dt/N2*1000
  RR = 1.0*len(net.layer[3].firings)/dt/N3*1000

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
  plt.plot(v[0])
  plt.subplot(221)
  plt.title('Left sensory neurons')
  plt.ylabel('Membrane potential (mV)')
  plt.ylim(-90, 40)

  plt.subplot(222)
  plt.plot(v[1])
  plt.title('Right sensory neurons')
  plt.ylim(-90, 40)

  plt.subplot(223)
  plt.plot(v[2])
  plt.title('Left motor neurons')
  plt.ylabel('Membrane potential (mV)')
  plt.ylim(-90, 40)
  plt.xlabel('Time (ms)')

  plt.subplot(224)
  plt.plot(v[3])
  plt.title('Right motor neurons')
  plt.ylim(-90, 40)
  plt.xlabel('Time (ms)')

  plt.draw()

  # Plot robot trajectory
  plt.figure(2)
  plt.scatter(x, y, marker='.')
  plt.draw()

