import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro

def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    delta0 = r2[0] - r1[0]
    delta1 = r2[1] - r1[1]
    delta2 = r2[2] - r1[2]
    delta0 -= L * round(delta0 / L)
    delta1 -= L * round(delta1 / L)
    delta2 -= L * round(delta2 / L)
    return (delta0 * delta0 + delta1 * delta1 + delta2 * delta2) ** 0.5

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
