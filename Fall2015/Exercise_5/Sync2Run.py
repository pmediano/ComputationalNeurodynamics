"""
Computational Neurodynamics
Exercise 5

(C) Murray Shanahan et al, 2015
"""

import sys
sys.path.append('../Exercise_2')

from Sync2Connect import Sync2Connect
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt


def Sync2Run():
  """
  Docstring
  """

  N1   = 800
  N2   = 200

  net = Sync2Connect(N1, N2)

  T  = 1000  # simulation time per episode
  Ib = 5     # base current

  # Initialise layers
  for lr in xrange(net.Nlayers):
    net.layer[lr].v = -65 * np.ones(net.layer[lr].N)
    net.layer[lr].u = net.layer[lr].b * net.layer[lr].v
    net.layer[lr].firings = np.array([])

  # SIMULATE
  for t in xrange(T):

    # Deliver base current
    net.layer[0].I = Ib*rn.randn(N1)
    net.layer[1].I = Ib*rn.randn(N2)
    # Delay the onset of activity for the second population
    if t > 12:
      net.layer[2].I = Ib*rn.randn(N1)
      net.layer[3].I = Ib*rn.randn(N2)
    else:
      net.layer[2].I = np.zeros(N1)
      net.layer[3].I = np.zeros(N2)

    ### Deliver a Poisson spike stream
    # lambda = 0.01
    # net.layer[1].I = 15*(poissrnd(lambda*15,N1) > 0)
    # net.layer[2].I = 15*(poissrnd(lambda*15,N2) > 0)

    # Update all the neurons
    net.Update(t)

  firings0 = net.layer[0].firings
  firings2 = net.layer[2].firings

  # Moving averages of firing rates in Hz for excitatory population
  ws = 10  # window size
  ds = 1   # slide window by ds
  MF0 = np.zeros(np.ceil(T*1.0/ds))
  MF2 = np.zeros(np.ceil(T*1.0/ds))

  for j in xrange(1, T, ds):
    MF0[j/ds] = sum(1*(firings0[:, 0] >= (j-ws)) * (firings0[:, 0] < j)) * 1000.0/(ws*N1)
    MF2[j/ds] = sum(1*(firings2[:, 0] >= (j-ws)) * (firings2[:, 0] < j)) * 1000.0/(ws*N1)

  # Raster plots of firings
  plt.subplot(211)
  if firings0.size is not 0:
    plt.scatter(firings0[:, 0], firings0[:, 1], marker='.')
  plt.xlim([0, T])
  plt.ylabel('Neuron number')
  plt.ylim([0, N1+1])

  plt.subplot(212)
  if firings2.size is not 0:
    plt.scatter(firings2[:, 0], firings2[:, 1], marker='.')
  plt.xlabel('Time (ms)')
  plt.xlim([0, T])
  plt.ylabel('Neuron number')
  plt.ylim([0, N1+1])

  # Plot mean firing rates
  plt.figure()
  plt.plot(range(0, T, ds), np.vstack([MF0, MF2]).T)
  plt.xlabel('Time (ms)')
  plt.ylabel('Mean firing rate')

  plt.show()

  return (MF0, MF2)

if __name__ == '__main__':
  Sync2Run()

