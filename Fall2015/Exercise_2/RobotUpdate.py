"""
Computational Neurodynamics
Exercise 2

(C) Murray Shanahan et al, 2015
"""

import numpy as np


def RobotUpdate(x1, y1, w1, UL, UR, Umax, dt, xmax, ymax):
  """
  Updates the position (x1,y1) and orientation w1 of the robot given
  wheel velocities UL (left) and UR (right), where Umax is the maximum
  wheel velocity, dt is the step size, and xmax and ymax are the limits of
  the torus.

  Outputs:
  x2, y2, w2 -- New position and orientation of the robot.
  """

  A = 1  # axle length

  BL = UL*Umax
  BR = UR*Umax
  B  = (BL + BR)/2.0
  C  = BR - BL
  dx = B*np.cos(w1)
  dy = B*np.sin(w1)
  dw = np.arctan2(C, A)

  x2 = x1 + dt*dx
  y2 = y1 + dt*dy
  w2 = w1 + dt*dw

  w2 = np.mod(w2+np.pi, 2*np.pi) - np.pi
  if w2 < 0:
    w2 = 2*np.pi + w2

  if x2 > xmax:
    x2 = x2 - xmax
  if y2 > ymax:
    y2 = y2 - ymax
  if x2 < 0:
    x2 = xmax + x2
  if y2 < 0:
    y2 = ymax + y2

  return x2, y2, w2

