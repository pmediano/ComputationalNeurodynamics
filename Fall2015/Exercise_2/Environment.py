"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import numpy.random as rn


class Environment:
  """
  Environment for the robot to run around. Holds a list of objects
  the robot should either avoid or catch.
  """

  def __init__(self, _Obs, _MinSize, _MaxSize, _xmax, _ymax):
    """
    Create a new environment comprising a list of length Obs of objects.
    Each new object is assigned a random size (between MinSize and MaxSize)
    and a random location on a torus. xmax and ymax are the limits of the torus

    Inputs:
    _Obs     -- Number of objects to draw in the environment
    _MinSize -- Minimum size of objects
    _MaxSize -- Maximum size of objects
    _xmax    -- Maximum X position of objects
    _ymax    -- Maximum Y position of objects
    """

    self.xmax = _xmax
    self.ymax = _ymax

    self.Obs = [{'x': rn.rand()*_xmax,
                 'y': rn.rand()*_ymax,
                 'r': _MinSize + rn.rand()*(_MaxSize - _MinSize)
                 } for _ in xrange(_Obs)
                ]

  def GetSensors(self, x, y, w):
    """
    Return the current activities of the robot's sensors given
    its position (x,y) and orientation w.
    All geometry is calculated on a torus with limits xmax and ymax.

    Inputs:
    x, y, w -- Position and orientation of robot

    Outputs:
    SL, SR -- Activities of left and right sensor
    """

    SL = 0
    SR = 0
    Range = 25.0  # Sensor range

    for Ob in self.Obs:
      x2 = Ob['x']
      y2 = Ob['y']

      # Find the shortest x distance on torus
      if abs(x2 + self.xmax - x) < abs(x2 - x):
        x2 = x2 + self.xmax
      elif abs(x2 - self.xmax - x) < abs(x2 - x):
        x2 = x2 - self.xmax

      # Find shortest y distance on torus
      if abs(y2 + self.ymax - y) < abs(y2 - y):
        y2 = y2 + self.ymax
      elif abs(y2 - self.ymax - y) < abs(y2 - y):
        y2 = y2 - self.ymax

      dx = x2 - x
      dy = y2 - y

      z = np.sqrt(dx**2 + dy**2)

      if z < Range:
        v = np.arctan2(dy, dx)
        if v < 0:
          v = 2*np.pi + v

        dw = v - w  # angle difference between robot's heading and object

        # Stimulus strength depends on distnace to object boundary
        S = (Range - z)/Range

        if ((dw >= np.pi/8 and dw < np.pi/2) or
                (dw < -1.5*np.pi and dw >= -2*np.pi+np.pi/8)):
          SL = max(S, SL)
          # SL += S
        elif ((dw > 1.5*np.pi and dw <= 2*np.pi - np.pi/8) or
                (dw <= -np.pi/8 and dw > -np.pi/2)):
          SR = max(S, SR)
          # SR += S

    return SL, SR

