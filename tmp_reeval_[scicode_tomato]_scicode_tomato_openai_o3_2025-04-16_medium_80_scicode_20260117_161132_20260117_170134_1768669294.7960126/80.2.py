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
def E_ij(r, sigma, epsilon, rc):
    '''Return the truncated-and-shifted Lennard-Jones (12-6) potential energy
    between a pair of particles separated by distance r.

    Parameters
    ----------
    r : float
        Separation between particles i and j.
    sigma : float
        Distance at which the Lennard-Jones potential is zero.
    epsilon : float
        Depth of the potential well.
    rc : float
        Cut-off distance: for r ≥ rc the potential is defined to be zero.

    Returns
    -------
    float
        Truncated and shifted Lennard-Jones potential energy E(r).
    '''
    if r <= 0.0:
        raise ValueError("Distance r must be positive.")
    if sigma <= 0.0 or rc <= 0.0:
        raise ValueError("sigma and rc must be positive.")
    if r >= rc:
        return 0.0
    inv_r = sigma / r
    inv_rc = sigma / rc
    inv_r6 = inv_r ** 6
    inv_rc6 = inv_rc ** 6
    E_r = 4.0 * epsilon * (inv_r6 * inv_r6 - inv_r6)
    E_rc = 4.0 * epsilon * (inv_rc6 * inv_rc6 - inv_rc6)
    E = E_r - E_rc
    return E

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
