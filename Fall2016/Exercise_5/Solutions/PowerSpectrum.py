"""
Computational Neurodynamics
Exercise 5

(C) Murray Shanahan et al, 2016
"""

from numpy.fft import rfft
import numpy as np
import matplotlib.pyplot as plt

def PowerSpectrum(X, Fs):
  """
  Calculate the power spectrum of real-valued signal X measured with a
  sampling frequency Fs. Result is in the same units as Fs.
  """

  X = X - X.mean()
  F = rfft(X)[:-1]
  freq = np.arange(0, 0.5*Fs, Fs*1.0/len(X))
  pw = np.abs(F)**2
  return freq, pw/max(pw)

