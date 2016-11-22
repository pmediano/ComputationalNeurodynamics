"""
Computational Neurodynamics
Exercise 5

(C) Murray Shanahan et al, 2016
"""

from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt


def SynchronisationIndex(MF1, MF2):
  """
  Computes the synchronisation index between two populations given firing data
  MF1 and MF2.
  """

  if len(MF1) != len(MF2):
    raise Exception("Both time series must have the same length")

  # Centre time series on zero
  MF1 = MF1 - np.mean(MF1)
  MF2 = MF2 - np.mean(MF2)

  # Calculate phase using Hilbert transform
  phase1 = np.angle(hilbert(MF1))
  phase2 = np.angle(hilbert(MF2))

  # Calculate synchronisation index
  phi = np.abs((np.exp(1j*phase1) + np.exp(1j*phase2)) / 2.0)

  return np.mean(phi)

