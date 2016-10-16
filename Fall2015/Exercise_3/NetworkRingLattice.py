"""
Computational Neurodynamics
Exercise 3

(C) Murray Shanahan et al, 2015
"""

import numpy as np


def NetworkRingLattice(N, k):
  """
  Creates a ring lattice with N nodes and neighbourhood size k.
  Choosing k = 2 connects each node to its nearest 2 nodes, k = 4
  to the nearest 4 and so on. Odd values of k are rounded down
  to the previous even number.

  You technically can choose k > N, but that will give you just a
  fully connected net.

  Inputs:
  N -- Number of nodes
  k -- Neighbourhood size of the initial ring lattice
  """

  # Create empty connectivity matrix
  CIJ = np.zeros([N, N])

  # Loop through the nodes and connect the neighbourhoods
  for i in range(N):
    # Note that since the network is undirected (symmetric) we only
    # need to look at the upper triangular bit of CIJ.
    for j in range(i+1, N):
      if i != j and min(abs(i-j), N - abs(i-j)) <= k/2.0:
        CIJ[i, j] = 1
        CIJ[j, i] = 1

  return(CIJ)

