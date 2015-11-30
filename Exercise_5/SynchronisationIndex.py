"""
Computational Neurodynamics
Exercise 5

(C) Murray Shanahan et al, 2015
"""

from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt


def SynchronisationIndex(MF1, MF2, N, T):
  """
  Computes the synchronisation index between two populations given firing data
  MF1 and MF2, where N is the total number of neurons in each population and T
  is the length of the run that produced the data

  """

  # Centre time series on zero
  MF1 = MF1 - np.mean(MF1)
  MF2 = MF2 - np.mean(MF2)

  # Discard initial transient period
  discard = 100
  MF1 = MF1[discard:]
  MF2 = MF2[discard:]

  # Calculate phase using Hilbert transform
  phase1 = np.angle(hilbert(MF1))
  phase2 = np.angle(hilbert(MF2))

  # Calculate synchronisation index
  phi = np.abs((np.exp(1j*phase1) + np.exp(1j*phase2)) / 2.0)

  print "Mean synchronisation: ", np.mean(phi)

  # Plot mean firing rates
  plt.subplot(311)
  plt.plot(range(T-discard), np.vstack([MF1, MF2]).T)
  plt.xlabel('Time (ms)')
  plt.ylabel('Mean firing rate')

  # Plot phase
  plt.subplot(312)
  plt.plot(range(T-discard), np.vstack([phase1, phase2]).T)
  plt.xlabel('Time (ms)')
  plt.ylabel('Phase')

  # Plot synchronisation index
  plt.subplot(313)
  plt.plot(range(T-discard), phi)
  plt.xlabel('Time (ms)')
  plt.ylabel('Sync. index')

  plt.show()

