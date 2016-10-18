"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2016
"""

import numpy as np
import numpy.random as rn
from IzNetwork import IzNetwork


def RobotConnect4L(Ns, Nm):
  """
  Construct four layers of Izhikevich neurons and connect them together.
  Layers 0 and 1 comprise sensory neurons, while layers 2 and 3 comprise
  motor neurons. Sensory neurons excite contralateral motor neurons causing
  seeking behaviour. Layers are heterogenous populations of Izhikevich
  neurons with slightly different parameter values.

  Inputs:
  Ns -- Number of neurons in sensory layers
  Nm -- Number of neurons in motor layers
  """

  F    = 50.0/np.sqrt(Ns)  # Scaling factor
  Dmax = 5                 # Maximum conduction delay

  N = 2*Ns + 2*Nm
  net = IzNetwork(N, Dmax)

  r = rn.rand(N)
  a = 0.02*np.ones(N)
  b = 0.2*np.ones(N)
  c = -65 + 15*(r**2)
  d = 8 - 6*(r**2)

  oneBlock   = np.ones((Ns, Nm))

  # Block [i,j] is the connection from layer i to layer j
  W = np.bmat([[np.zeros((Ns,Ns)), np.zeros((Ns, Ns)), np.zeros((Ns,Nm)),        F*oneBlock],
               [np.zeros((Ns,Ns)), np.zeros((Ns, Ns)),        F*oneBlock, np.zeros((Ns,Nm))],
               [np.zeros((Ns,Nm)), np.zeros((Ns, Nm)), np.zeros((Nm,Nm)), np.zeros((Nm,Nm))],
               [np.zeros((Ns,Nm)), np.zeros((Ns, Nm)), np.zeros((Nm,Nm)), np.zeros((Nm,Nm))]])

  D = Dmax*np.ones((N,N), dtype=int)

  net.setParameters(a, b, c, d)
  net.setDelays(D)
  net.setWeights(W)

  return net

