"""
Computational Neurodynamics
Exercise 2

Simulates the movement of a robot with differential wheels under the
control of a spiking neural network. The simulation runs for a very
long time --- if you get bored, press Ctrl+C a couple of times.

(C) Murray Shanahan et al, 2016
"""

import numpy as np
import numpy.random as rn
import pylab
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

# Set voltage axes
fig1 = pylab.figure(1)
ax11 = fig1.add_subplot(221)
ax12 = fig1.add_subplot(222)
ax21 = fig1.add_subplot(223)
ax22 = fig1.add_subplot(224)

pl11 = ax11.plot(v[:,0:N1])
ax11.set_title('Left sensory neurons')
ax11.set_ylabel('Membrane potential (mV)')
ax11.set_ylim(-90, 40)

pl12 = ax12.plot(v[:,N1:(N1+N2)])
ax12.set_title('Right sensory neurons')
ax12.set_ylim(-90, 40)

pl21 = ax21.plot(v[:,(N1+N2):(N1+N2+N3)])
ax21.set_title('Left motor neurons')
ax21.set_ylabel('Membrane potential (mV)')
ax21.set_ylim(-90, 40)
ax21.set_xlabel('Time (ms)')

pl22 = ax22.plot(v[:,(N-N4):N])
ax22.set_title('Right motor neurons')
ax22.set_ylim(-90, 40)
ax22.set_xlabel('Time (ms)')

manager1 = pylab.get_current_fig_manager()

# Draw Environment
fig2 = pylab.figure(2)
ax2 = fig2.add_subplot(111)
ax2.axis([0, xmax, 0, ymax])
ax2.set_title('Robot controlled by spiking neurons')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
for Ob in Env.Obs:
  ax2.scatter(Ob['x'], Ob['y'], s=np.pi*(Ob['r']**2), c='lime')

manager2 = pylab.get_current_fig_manager()

# You can change the interval duration to make the video faster or slower
timer = fig2.canvas.new_timer(interval=200)

def StopSimulation():
  global timer
  timer.stop()

t = 0

## SIMULATE
print 'Start simulation'
def RobotStep(args):
  global t

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
  for i in range(Ns):
    pl11[i].set_data(range(dt), v[:,i])
    pl12[i].set_data(range(dt), v[:,i+Ns])

  for i in range(Nm):
    pl21[i].set_data(range(dt), v[:,2*Ns+i])
    pl22[i].set_data(range(dt), v[:,2*Ns+Nm+i])

  ax2.scatter(x, y)
  manager1.canvas.draw()
  manager2.canvas.draw()

  t += 1

  if t == len(x)-1:
    print 'Terminating simulation'
    StopSimulation()

# Get the thing going
timer.add_callback(RobotStep, ())
timer.start()

pylab.show()

