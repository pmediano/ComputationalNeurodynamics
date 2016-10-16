import numpy as np


class IzNetwork:
  """
  Network of Izhikevich neurons.
  """

  def __init__(self, _neuronsPerLayer, _Dmax):
    """
    Initialise network with given number of neurons

    Inputs:
    _neuronsPerLayer -- List with the number of neurons in each layer. A list
                        [N1, N2, ... Nk] will return a network with k layers
                        with the corresponding number of neurons in each.

    _Dmax            -- Maximum delay in all the synapses in the network. Any
                        longer delay will result in failing to deliver spikes.
    """

    self.Dmax = _Dmax
    self.Nlayers = len(_neuronsPerLayer)

    self.layer = {}

    for i, n in enumerate(_neuronsPerLayer):
      self.layer[i] = IzLayer(n)

  def Update(self, t):
    """
    Run simulation of the whole network for 1 millisecond and update the
    network's internal variables.

    Inputs:
    t -- Current timestep. Necessary to sort out the synaptic delays.
    """
    for lr in xrange(self.Nlayers):
      self.NeuronUpdate(lr, t)

  def NeuronUpdate(self, i, t):
    """
    Izhikevich neuron update function. Update one layer for 1 millisecond
    using the Euler method.

    Inputs:
    i -- Number of layer to update
    t -- Current timestep. Necessary to sort out the synaptic delays.
    """

    # Euler method step size in ms
    dt = 0.2

    # Calculate current from incoming spikes
    for j in xrange(self.Nlayers):

      # If layer[i].S[j] exists then layer[i].factor[j] and
      # layer[i].delay[j] have to exist
      if j in self.layer[i].S:
        S = self.layer[i].S[j]  # target neuron->rows, source neuron->columns

        # Firings contains time and neuron idx of each spike.
        # [t, index of the neuron in the layer j]
        firings = self.layer[j].firings

        # Find incoming spikes taking delays into account
        delay = self.layer[i].delay[j]
        F = self.layer[i].factor[j]

        # Sum current from incoming spikes
        k = len(firings)
        while k > 0 and (firings[k-1, 0] > (t - self.Dmax)):
          idx = delay[:, firings[k-1, 1]] == (t-firings[k-1, 0])
          self.layer[i].I[idx] += F * S[idx, firings[k-1, 1]]
          k = k-1

    # Update v and u using the Izhikevich model and Euler method
    for k in xrange(int(1/dt)):
      v = self.layer[i].v
      u = self.layer[i].u

      self.layer[i].v += dt*(0.04*v*v + 5*v + 140 - u + self.layer[i].I)
      self.layer[i].u += dt*(self.layer[i].a*(self.layer[i].b*v - u))

      # Find index of neurons that have fired this millisecond
      fired = np.where(self.layer[i].v >= 30)[0]

      if len(fired) > 0:
        for f in fired:
          # Add spikes into spike train
          if len(self.layer[i].firings) != 0:
            self.layer[i].firings = np.vstack([self.layer[i].firings, [t, f]])
          else:
            self.layer[i].firings = np.array([[t, f]])

          # Reset the membrane potential after spikes
          self.layer[i].v[f]  = self.layer[i].c[f]
          self.layer[i].u[f] += self.layer[i].d[f]

    return


class IzLayer:
  """
  Layer of Izhikevich neurons to be used inside an IzNetwork.
  """

  def __init__(self, n):
    """
    Initialise layer with empty vectors.

    Inputs:
    n -- Number of neurons in the layer
    """

    self.N = n
    self.a = np.zeros(n)
    self.b = np.zeros(n)
    self.c = np.zeros(n)
    self.d = np.zeros(n)

    self.S      = {}
    self.delay  = {}
    self.factor = {}

