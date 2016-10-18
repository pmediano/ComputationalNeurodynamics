"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2016
"""

import numpy as np

class IzNetwork(object):
  """
  This class is used to simulate a network of Izhikevich neurons. The state of
  the neurons is automatically initialised, and parameters and connectivity
  matrices can be set with the appropriate setter methods. All class members are
  hidden (i.e. underscored) except for the state of the neurons (v,u).
  
  For both the delay and weight connectivity matrices, A[i,j] refers to the
  connection from neuron i to j. This was done this way (against standard
  convention) because the algorithm is easier to vectorise this way.
  
  Vectorisation with inhomogeneous time-delays is accomplished via a cylindrical
  accumulator array X, that is updated at every time step. More details in the
  inline comments.
  
  References:

  Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions
  on Neural Networks, 14(6), 1569-72. http://doi.org/10.1109/TNN.2003.820440

  Brette, R., & Goodman, D. F. M. (2011). Vectorized algorithms for spiking
  neural network simulation. Neural Computation, 23(6), 1503-35.
  http://doi.org/10.1162/NECO_a_00123

  """

  def __init__(self, N, Dmax):
    """
    Initialise network with given number of neurons and maximum transmission
    delay.

    Inputs:
    N     -- Number of neurons in the network.

    Dmax  -- Maximum delay in all the synapses in the network, in ms. Any
             longer delay will result in failing to deliver spikes.
    """
    self._Dmax   = Dmax + 1
    self._N      = N
    self._X      = np.zeros((Dmax + 1, N))
    self._I      = np.zeros(N)
    self._cursor = 0
    self._lastFired = np.array([False]*N)
    self._dt     = 0.1
    self.v       = -65.0*np.ones(N)
    self.u       = -1.0*np.ones(N)


  def setDelays(self, D):
    """
    Set synaptic delays.
    
    Inputs:
    D  -- np.array or np.matrix. The delay matrix must contain nonnegative
          integers, and must be of size N-by-N, where N is the number of
          neurons supplied in the constructor.
    """
    if D.shape != (self._N, self._N):
      raise Exception('Delay matrix must be N-by-N.')

    if not np.issubdtype(D.dtype, np.integer):
      raise Exception('Delays must be integer numbers.')

    if (D < 0.5).any():
      raise Exception('Delays must be strictly positive.')

    self._D = D


  def setWeights(self, W):
    """
    Set synaptic weights.

    Inputs:
    W  -- np.array or np.matrix. The weight matrix must be of size N-by-N,
          where N is the number of neurons supplied in the constructor.
    """
    if W.shape != (self._N, self._N):
      raise Exception('Weight matrix must be N-by-N.')
    self._W = W


  def setCurrent(self, I):
    """
    Set the external current input to the network for this timestep. This
    only affects the next call to update().

    Inputs:
    I  -- np.array. Must be of length N, where N is the number of neurons
          supplied in the constructor.
    """
    if len(I) != self._N:
      raise Exception('Current vector must be of size N.')
    self._I = I


  def setParameters(self, a, b, c, d):
    """
    Set parameters for the neurons. Names are the the same as in Izhikevich's
    original paper, see references above. All inputs must be np.arrays of size
    N, where N is the number of neurons supplied in the constructor.
    """
    if (len(a), len(b), len(c), len(d)) != (self._N, self._N, self._N, self._N):
      raise Exception('Parameter vectors must be of size N.')

    self._a = a
    self._b = b
    self._c = c
    self._d = d


  def getState(self):
    """
    Get current state of the network. Outputs a tuple with two np.arrays,
    corresponding to the V and the U of the neurons in the network in this
    timestep.
    """
    return (self.v, self.u)


  def update(self):
    """
    Simulate one millisecond of network activity. The internal dynamics
    of each neuron are simulated using the Euler method with step size
    self._dt, and spikes are delivered every millisecond.

    Returns the indices of the neurons that fired this millisecond.
    """

    # Reset neurons that fired last timestep
    self.v[self._lastFired]  = self._c[self._lastFired]
    self.u[self._lastFired] += self._d[self._lastFired]

    # Input current is the sum of external and internal contributions
    I = self._I + self._X[self._cursor%self._Dmax,:]

    # Update v and u using the Izhikevich model and Euler method. To avoid
    # overflows with large input currents, keep updating only neurons that
    # haven't fired this millisecond.
    fired = np.array([False]*self._N)
    for _ in range(int(1/self._dt)):
        notFired = np.logical_not(fired)
        v = self.v[notFired]
        u = self.u[notFired]
        self.v[notFired] += self._dt*(0.04*v*v + 5*v + 140 - u + I[notFired])
        self.u[notFired] += self._dt*(self._a[notFired]*(self._b[notFired]*v - u))
        fired = np.logical_or(fired, self.v > 30)

    # Find which neurons fired this timestep. Their membrane potential is
    # fixed at 30 for visualisation purposes, and will be reset according to
    # the Izhikevich equation in the next iteration
    fired_idx = np.where(fired)[0]
    self._lastFired = fired
    self.v[fired]   = 30*np.ones(len(fired_idx))

    # Clear current for next timestep
    self._I = np.zeros(self._N)

    # Here's where the magic happens. For each firing "source" neuron i and
    # each "target" neuron j, we add a contribution W[i,j] to the accumulator
    # D[i,j] timesteps into the future. That way, as the cursor moves X
    # contains all the input coming from time-delayed connections.
    for i in fired_idx:
      self._X[(self._cursor + self._D[i, :])%self._Dmax, :] += self._W[i,:]

    # Increment the cursor for the cylindrical array and clear accumulator
    self._X[self._cursor%self._Dmax,:] = np.zeros(self._N)
    self._cursor += 1

    return fired_idx

