"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

from IzNetwork import IzNetwork
import numpy as np
import numpy.random as rn


def Connect2L(N0, N1):
  """
  Constructs two layers of Izhikevich neurons and connects them together.
  Layers are arrays of N neurons. Parameters for regular spiking neurons
  extracted from:

  http://www.izhikevich.org/publications/spikes.htm
  """

  F = 50/np.sqrt(N1)  # Scaling factor
  D = 5               # Conduction delay
  Dmax = 10           # Maximum conduction delay

  net = IzNetwork([N0, N1], Dmax)

  # Neuron parameters
  # Each layer comprises a heterogenous set of neurons, with a small spread
  # of parameter values, so that they exhibit some dynamical variation
  # (To get a homogenous population of canonical "regular spiking" neurons,
  # multiply r by zero.)

  # Layer 0 (regular spiking)
  r = rn.rand(N0)
  net.layer[0].N = N0
  net.layer[0].a = 0.02 * np.ones(N0)
  net.layer[0].b = 0.20 * np.ones(N0)
  net.layer[0].c = -65 + 15*(r**2)
  net.layer[0].d = 8 - 6*(r**2)

  # Layer 1 (regular spiking)
  r = rn.rand(N1)
  net.layer[1].N = N1
  net.layer[1].a = 0.02 * np.ones(N1)
  net.layer[1].b = 0.20 * np.ones(N1)
  net.layer[1].c = -65 + 15*(r**2)
  net.layer[1].d = 8 - 6*(r**2)

  ## Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # S(i,j) is the strength of the connection from neuron j to neuron i
  net.layer[1].S[0]      = np.ones([N1, N0])
  net.layer[1].factor[0] = F
  net.layer[1].delay[0]  = D * np.ones([N1, N0], dtype=int)

  return net

