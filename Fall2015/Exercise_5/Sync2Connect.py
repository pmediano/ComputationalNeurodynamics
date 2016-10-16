"""
Computational Neurodynamics
Exercise 5

(C) Murray Shanahan et al, 2015
"""

import sys
sys.path.append('../Exercise_2')

from IzNetwork import IzNetwork
import numpy as np
import numpy.random as rn


def Sync2Connect(N1, N2):
  """
  Constructs two populations of neurons (comprising two layers each) that
  oscillate in the gamma range using PING, and that can be couped together to
  study various synchronisation phenomena

  Coupling the excitatory populations causes the populations to synhronise 180
  degrees out of phase with each other if they have the same natural frequency.
  Coupling the inhibitory populations causes complete synchronisation with zero
  phase lag, even if they have slightly different natural frequencies.
  """

  # Conduction delays - 2 for 56Hz, 5 for 40Hz, 8 for 32Hz
  D1 = 6
  D2 = 4
  D = 5  # conduction delay for inter-population connections

  net = IzNetwork([N1, N2, N1, N2], max(D1, D2) + 1)

  # Neuron parameters
  # Each layer comprises a heterogenous set of neurons, with a small spread
  # of parameter values, so that they exhibit some dynamical variation

  # Layer 0 (excitatory - regular spiking)
  r = rn.rand(N1)
  net.layer[0].N = N1
  net.layer[0].a = 0.02 * np.ones(N1)
  net.layer[0].b = 0.20 * np.ones(N1)
  net.layer[0].c = -65 + 15*(r**2)
  net.layer[0].d = 8 - 6*(r**2)

  # Layer 1 (inhibitory - fast spiking)
  r = rn.rand(N2)
  net.layer[1].N = N2
  net.layer[1].a = 0.02 + 0.08*r
  net.layer[1].b = 0.25 - 0.05*r
  net.layer[1].c = -65 * np.ones(N2)
  net.layer[1].d = 2 * np.ones(N2)

  # Layer 2 (excitatory - regular spiking)
  r = rn.rand(N1)
  net.layer[2].N = N1
  net.layer[2].a = 0.02 * np.ones(N1)
  net.layer[2].b = 0.20 * np.ones(N1)
  net.layer[2].c = -65 + 15*(r**2)
  net.layer[2].d = 8 - 6*(r**2)

  # Layer 3 (inhibitory - fast spiking)
  r = rn.rand(N2)
  net.layer[3].N = N2
  net.layer[3].a = 0.02 + 0.08*r
  net.layer[3].b = 0.25 - 0.05*r
  net.layer[3].c = -65 * np.ones(N2)
  net.layer[3].d = 2 * np.ones(N2)

  # Connectivity matrix (synaptic weights)
  # layer{i}.S{j} is the connectivity matrix from layer j to layer i
  # s(i,j) is the strength of the connection from neuron j to neuron i

  # Excitatory to inhibitory connections
  net.layer[1].S[0] = rn.rand(N2, N1)
  net.layer[3].S[2] = rn.rand(N2, N1)

  # Inhibitory to excitatory connections
  net.layer[0].S[1] = -1.0*rn.rand(N1, N2)
  net.layer[2].S[3] = -1.0*rn.rand(N1, N2)

  # Excitatory to excitatory connections
  net.layer[0].S[0] = 1*(rn.rand(N1, N1) < 0.01)
  net.layer[2].S[2] = 1*(rn.rand(N1, N1) < 0.01)

  # Inhibitory to inhibitory connections
  net.layer[1].S[1] = -1.0*rn.rand(N2, N2)
  net.layer[3].S[3] = -1.0*rn.rand(N2, N2)

  # Coupling between populations (excitatory to excitatory)
  net.layer[2].S[0] = 1*(rn.rand(N1, N1) < 0.01)
  net.layer[0].S[2] = 1*(rn.rand(N1, N1) < 0.01)

  # Coupling between populations (excitatory to inhibitory)
  net.layer[3].S[0] = 1*(rn.rand(N2, N1) < 0.01)
  net.layer[1].S[2] = 1*(rn.rand(N2, N1) < 0.01)

  ## Scaling factors
  # Within oscillator 1
  net.layer[1].factor[0] = 2
  net.layer[0].factor[1] = 2
  net.layer[0].factor[0] = 17  # use 5 if excitatory couplings exist
  net.layer[1].factor[1] = 2

  # Within oscillator 2
  net.layer[3].factor[2] = 2
  net.layer[2].factor[3] = 2
  net.layer[2].factor[2] = 17  # use 5 if excitatory couplings exist
  net.layer[3].factor[3] = 2

  # Excit-Excit coupling. Deactivated by default
  net.layer[2].factor[0] = 0
  net.layer[0].factor[2] = 0

  # Excit-Inhib coupling
  net.layer[3].factor[0] = 12
  net.layer[1].factor[2] = 12

  ## Conduction delays
  # Within oscillator 1
  net.layer[1].delay[0] = D1 * np.ones([N2, N1])
  net.layer[0].delay[1] = D1 * np.ones([N1, N2])
  net.layer[0].delay[0] = D1 * np.ones([N1, N1])
  net.layer[1].delay[1] = D1 * np.ones([N2, N2])

  # Within oscillator 2
  net.layer[3].delay[2] = D2 * np.ones([N2, N1])
  net.layer[2].delay[3] = D2 * np.ones([N1, N2])
  net.layer[2].delay[2] = D2 * np.ones([N1, N1])
  net.layer[3].delay[3] = D2 * np.ones([N2, N2])

  net.layer[2].delay[0] = D * np.ones([N1, N1])
  net.layer[0].delay[2] = D * np.ones([N1, N1])

  net.layer[3].delay[0] = D * np.ones([N2, N1])
  net.layer[1].delay[2] = D * np.ones([N2, N1])

  return net

