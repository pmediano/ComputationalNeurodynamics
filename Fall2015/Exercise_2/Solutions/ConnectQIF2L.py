"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

from QIFNetwork import QIFNetwork
import numpy as np
import numpy.random as rn


def ConnectQIF2L(N0, N1):
  """
  Constructs two layers of QIF neurons and connects them together.  Pretty much
  like Connect2L, but with QIF instead of Izhikevich neurons. Layers are
  arrays of N neurons.

  Inputs:
  N0, N1 -- Number of neurons in layer 0 and 1, respectively
  """

  F = 60/np.sqrt(N1)  # Scaling factor
  D = 5               # Conduction delay
  Dmax = 10           # Maximum conduction delay

  net = QIFNetwork([N0, N1], Dmax)

  # Neuron parameters
  # Each layer comprises a heterogenous set of neurons, with a small spread
  # of parameter values, so that they exhibit some dynamical variation

  # Layer 0
  r = rn.rand(N0)
  net.layer[0].N = N0
  net.layer[0].R = 1.0
  net.layer[0].tau = 10
  net.layer[0].vr = -65 + 10*(r**2)
  net.layer[0].vc = -50 + 5*(r**2)
  net.layer[0].a = 0.2

  # Layer 1
  r = rn.rand(N1)
  net.layer[1].N = N1
  net.layer[1].R = 1.0
  net.layer[1].tau = 5
  net.layer[1].vr = -65 + 10*(r**2)
  net.layer[1].vc = -50 + 5*(r**2)
  net.layer[1].a = 0.2

  ## Connectivity matrix (synaptic weights)
  # layer[i].S[j] is the connectivity matrix from layer j to layer i
  # S(i,j) is the strength of the connection from neuron j to neuron i
  net.layer[1].S[0]      = np.ones([N1, N0])
  net.layer[1].factor[0] = F
  net.layer[1].delay[0]  = D * np.ones([N1, N0], dtype=int)

  return net

