"""
Computational Neurodynamics
Exercise 3

Randomly generates networks according to the Watts-Strogatz procedure,
computes their small-world indices, and local and global efficiencies,
and plots them. N is the number of nodes, k is the neighbourhood size.

(C) Murray Shanahan et al, 2015
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import bct
from SmallWorldIndex import SmallWorldIndex
from NetworkWattsStrogatz import NetworkWattsStrogatz

# Set up parameter values
nb_trials = 100
N = 200
k = 4

# Pre-allocate arrays
prob = 10**(-3*np.random.random(nb_trials))
SWI   = np.zeros(nb_trials)
Eglob = np.zeros(nb_trials)
Eloc  = np.zeros(nb_trials)

# Calculate statistics
for i, p in enumerate(prob):
  CIJ = NetworkWattsStrogatz(N, k, p)

  SWI[i]   = SmallWorldIndex(CIJ)
  Eglob[i] = bct.efficiency_wei(CIJ, local=False)
  Eloc[i]  = np.mean(bct.efficiency_wei(CIJ, local=True))

# Plot figures
plt.figure(1)
plt.semilogx(prob, SWI, marker='.', linestyle='none')
plt.xlabel('Rewiring probability')
plt.ylabel('Small World Index')

plt.figure(2)
plt.subplot(211)
plt.semilogx(prob, Eglob, marker='.', linestyle='none')
plt.xlabel('Rewiring probability')
plt.ylabel('Global efficiency')
plt.subplot(212)
plt.semilogx(prob, Eloc, marker='.', linestyle='none')
plt.xlabel('Rewiring probability')
plt.ylabel('Local efficiency')
plt.show()

