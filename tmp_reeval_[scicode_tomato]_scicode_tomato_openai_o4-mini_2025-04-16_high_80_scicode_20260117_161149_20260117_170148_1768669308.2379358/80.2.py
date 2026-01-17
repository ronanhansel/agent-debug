import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro

def dist(r1, r2, L):
    """Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : array-like of length 3
    r2 : array-like of length 3
    L  : float, side length of the cubic box
    Returns:
    float: the minimum-image Euclidean distance between r1 and r2
    """
    r1 = [float(x) for x in r1]
    r2 = [float(x) for x in r2]
    delta = [a - b for a, b in zip(r1, r2)]
    delta = [d - L * round(d / L) for d in delta]
    return sum(d * d for d in delta) ** 0.5
def E_ij(r, sigma, epsilon, rc):
    """
    Calculate the truncated and shifted Lennard-Jones potential energy between two particles.

    Parameters:
    r (float): Distance between particles i and j.
    sigma (float): Distance at which the Lennard-Jones potential is zero.
    epsilon (float): Depth of the potential well.
    rc (float): Cutoff distance beyond which the potential is zero.

    Returns:
    float: Truncated and shifted Lennard-Jones potential energy.
    """
    if r >= rc:
        return 0.0
    v_rc = 4.0 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
    v_r = 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    return v_r - v_rc

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.2', 3)
target = targets[0]

r1 = 1.0  # Close to the sigma value
sigma1 = 1.0
epsilon1 = 1.0
rc = 1
assert np.allclose(E_ij(r1, sigma1, epsilon1, rc), target)
target = targets[1]

r2 = 0.5  # Significantly closer than the effective diameter
sigma2 = 1.0
epsilon2 = 1.0
rc = 2
assert np.allclose(E_ij(r2, sigma2, epsilon2, rc), target)
target = targets[2]

r3 = 2.0  # Larger than sigma
sigma3 = 1.0
epsilon3 = 1.0
rc = 5
assert np.allclose(E_ij(r3, sigma3, epsilon3, rc), target)
