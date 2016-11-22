"""
Computational Neurodynamics
Exercise 5

Run an example with two coupled neural oscillators.

(C) Murray Shanahan et al, 2016
"""

import sys
sys.path.append('../Exercise_2')

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from SynchronisationIndex import SynchronisationIndex
from PowerSpectrum import PowerSpectrum
import IzNetwork as iz


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

  # Conduction delays - 6 for 34Hz, 4 for 44Hz approx
  D1 = 6
  D2 = 4
  D  = 5  # conduction delay for inter-population connections

  net = iz.IzNetwork(2*N1 + 2*N2, max(D1, D2) + 1)

  # Neuron parameters
  # Each layer comprises a heterogenous set of neurons, with a small spread
  # of parameter values, so that they exhibit some dynamical variation

  # Excitatory layer 1 (regular spiking)
  r = rn.rand(N1)
  ex1_a = 0.02 * np.ones(N1)
  ex1_b = 0.20 * np.ones(N1)
  ex1_c = -65 + 15*(r**2)
  ex1_d = 8 - 6*(r**2)

  # Inhibitory layer 1 (fast spiking)
  r = rn.rand(N2)
  in1_a = 0.02 + 0.08*r
  in1_b = 0.25 - 0.05*r
  in1_c = -65 * np.ones(N2)
  in1_d = 2 * np.ones(N2)

  # Excitatory layer 2 (regular spiking)
  r = rn.rand(N1)
  ex2_a = 0.02 * np.ones(N1)
  ex2_b = 0.20 * np.ones(N1)
  ex2_c = -65 + 15*(r**2)
  ex2_d = 8 - 6*(r**2)

  # Inhibitory layer 2 (fast spiking)
  r = rn.rand(N2)
  in2_a = 0.02 + 0.08*r
  in2_b = 0.25 - 0.05*r
  in2_c = -65 * np.ones(N2)
  in2_d = 2 * np.ones(N2)

  # All layers
  a = np.hstack([ex1_a, in1_a, ex2_a, in2_a])
  b = np.hstack([ex1_b, in1_b, ex2_b, in2_b])
  c = np.hstack([ex1_c, in1_c, ex2_c, in2_c])
  d = np.hstack([ex1_d, in1_d, ex2_d, in2_d])

  ## Connectivity matrix (synaptic weights)
  sparseSynapse = lambda n,m: 1*(rn.rand(n, m) < 0.01)
  denseSynapse  = lambda n,m: rn.rand(n, m)

  W = np.bmat([
      [17*sparseSynapse(N1,N1),   2*denseSynapse(N1,N2),       np.zeros((N1,N1)),       np.zeros((N1,N2))],
      [ -2*denseSynapse(N2,N1),  -2*denseSynapse(N2,N2),       np.zeros((N2,N1)),       np.zeros((N2,N2))],
      [      np.zeros((N1,N1)),       np.zeros((N1,N2)), 17*sparseSynapse(N1,N1),   2*denseSynapse(N1,N2)],
      [      np.zeros((N2,N1)),       np.zeros((N2,N2)),  -2*denseSynapse(N2,N1),  -2*denseSynapse(N2,N2)]
    ])

  ## Ex-In coupled PING oscillators. This should produce synchronous gamma oscillations.
  # W = np.bmat([
  #     [17*sparseSynapse(N1,N1),   2*denseSynapse(N1,N2),       np.zeros((N1,N1)), 12*sparseSynapse(N1,N2)],
  #     [ -2*denseSynapse(N2,N1),  -2*denseSynapse(N2,N2),       np.zeros((N2,N1)),       np.zeros((N2,N2))],
  #     [      np.zeros((N1,N1)), 12*sparseSynapse(N1,N2), 17*sparseSynapse(N1,N1),   2*denseSynapse(N1,N2)],
  #     [      np.zeros((N2,N1)),       np.zeros((N2,N2)),  -2*denseSynapse(N2,N1),  -2*denseSynapse(N2,N2)]
  #   ])

  ## Ex-Ex coupled PING oscillators. This should produce a coupled theta-gamma rhythm.
  # W = np.bmat([
  #     [ 5*sparseSynapse(N1,N1),   2*denseSynapse(N1,N2),  5*sparseSynapse(N1,N1),       np.zeros((N1,N2))],
  #     [ -2*denseSynapse(N2,N1),  -2*denseSynapse(N2,N2),       np.zeros((N2,N1)),       np.zeros((N2,N2))],
  #     [ 5*sparseSynapse(N1,N1),       np.zeros((N1,N2)),  5*sparseSynapse(N1,N1),   2*denseSynapse(N1,N2)],
  #     [      np.zeros((N2,N1)),       np.zeros((N2,N2)),  -2*denseSynapse(N2,N1),  -2*denseSynapse(N2,N2)]
  #   ])

  ## Conduction delays
  D = np.bmat([[D1*np.ones((N1+N2,N1+N2)),  D*np.ones((N1+N2,N1+N2))],
               [ D*np.ones((N1+N2,N1+N2)), D2*np.ones((N1+N2,N1+N2))]])
  D = np.array(D, dtype=int)

  # In the theta-gamma case, change all delays to 5ms
  # D = D*np.ones((N,N), dtype=int)

  net.setWeights(W)
  net.setDelays(D)
  net.setParameters(a, b, c, d)

  return net


def Sync2Run(N1, N2, T):
  """
  Simulate two coupled PING oscillators for T milliseconds. Each oscillator
  has N1 excitatory neurons and N2 inhibitory neurons.

  Returns the full history of spikes in the network as a Mx2 matrix, where
  each row [spike_time, neuron_nb] represents a single spike.
  """

  N  = 2*N1 + 2*N2
  S = np.array([])

  net = Sync2Connect(N1, N2)

  Ib = 5   # base current
  Ip = 15  # peak current

  # SIMULATE
  for t in xrange(T):

    # Deliver current
    I = Ip*(rn.rand(N) < 0.01) + Ib*rn.rand(N)

    # Update all the neurons
    net.setCurrent(I)
    spikes = net.update()
    if len(S) < 1:
      S = np.vstack([t*np.ones(len(spikes)), spikes]).T
    else:
      S = np.vstack([S, np.vstack([t*np.ones(len(spikes)), spikes]).T])

  return S


if __name__ == '__main__':
  """
  Create and run network.
  """

  T  = 1000
  N1 = 800
  N2 = 200
  N  = 2*N1 + 2*N2

  # Run simulation and get history of spikes
  S = Sync2Run(N1, N2, T)

  # Get firings of both excitatory layers
  firings0 = S[S[:,1] < N1]
  firings2 = S[np.logical_and(S[:,1] > N1+N2, S[:,1] < N-N2)]

  # Moving averages of firing rates in Hz for excitatory populations
  ws = 10  # window size
  ds = 1   # slide window by ds
  MF0 = np.zeros(int(np.ceil(T*1.0/ds)))
  MF2 = np.zeros(int(np.ceil(T*1.0/ds)))

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
    plt.scatter(firings2[:, 0], firings2[:, 1] - (N1+N2), marker='.')
  plt.xlabel('Time (ms)')
  plt.xlim([0, T])
  plt.ylabel('Neuron number')
  plt.ylim([0, N1+1])

  # Plot mean firing rates
  plt.figure()
  plt.plot(range(0, T, ds), np.vstack([MF0, MF2]).T)
  plt.xlabel('Time (ms)')
  plt.ylabel('Mean firing rate')

  # Plot power spectrum
  plt.figure()
  plt.subplot(211)
  freq, pw = PowerSpectrum(MF0[100:], 1000/ds)
  plt.plot(freq, pw)
  plt.ylabel('Spectral power')
  plt.xlim([0, 100])  # Just for visualisation purposes
  plt.subplot(212)
  freq, pw = PowerSpectrum(MF2[100:], 1000/ds)
  plt.plot(freq, pw)
  plt.xlim([0, 100])  # Just for visualisation purposes
  plt.ylabel('Spectral power')
  plt.xlabel('Frequency (Hz)')

  phi = SynchronisationIndex(MF0, MF2)
  print "Mean synchronisation: ", phi

  plt.show()

