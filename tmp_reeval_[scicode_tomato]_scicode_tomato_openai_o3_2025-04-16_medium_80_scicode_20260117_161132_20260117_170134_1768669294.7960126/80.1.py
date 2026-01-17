import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro

def dist(r1, r2, L):
    '''Calculate the minimum-image distance between two atoms in a periodic cubic system.

    Parameters
    ----------
    r1 : sequence of 3 floats
        (x, y, z) coordinates of the first atom.
    r2 : sequence of 3 floats
        (x, y, z) coordinates of the second atom.
    L : float
        Length of the cubic box edge.

    Returns
    -------
    float
        Minimum-image distance between the two atoms.
    '''
    x1, y1, z1 = map(float, r1)
    x2, y2, z2 = map(float, r2)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    return (dx * dx + dy * dy + dz * dz) ** 0.5

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.1', 3)
target = targets[0]

r1 = np.array([2.0, 3.0, 4.0])
r2 = np.array([2.5, 3.5, 4.5])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
target = targets[1]

r1 = np.array([1.0, 1.0, 1.0])
r2 = np.array([9.0, 9.0, 9.0])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
target = targets[2]

r1 = np.array([0.1, 0.1, 0.1])
r2 = np.array([9.9, 9.9, 9.9])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
