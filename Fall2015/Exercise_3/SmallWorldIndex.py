"""
Computational Neurodynamics
Exercise 3

(C) Murray Shanahan et al, 2015
"""

import numpy as np
from bct import breadthdist, charpath, clustering_coef_bu


def SmallWorldIndex(CIJ):
  """
  Computes the small-world index of the graph with connection matrix CIJ.
  Self-connections are ignored, as they are cyclic paths.

  Inputs:
  CIJ  --  Graph connectivity matrix. Must be binary (0 or 1) and undirected.
  """

  N = len(CIJ)
  K = np.sum(np.sum(CIJ))/len(CIJ)  # average degree

  # Clustering coefficient
  CC = np.mean(clustering_coef_bu(CIJ))

  # Distance matrix
  [RR, DD] = breadthdist(CIJ)

  # Note: the charpath implementation of bctpy is not very robust. Expect
  # some warnings. From the output of charpath use only the characteristic
  # path length
  PL = charpath(DD, include_diagonal=False)[0]

  # Calculate small-world index
  CCs = CC/(K/N)
  PLs = PL/(np.log(N)/np.log(K))
  SWI = CCs/PLs

  return(SWI)

