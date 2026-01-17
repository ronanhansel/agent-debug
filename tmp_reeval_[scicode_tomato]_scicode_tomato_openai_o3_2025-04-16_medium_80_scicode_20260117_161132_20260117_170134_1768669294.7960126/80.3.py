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
def E_pot(xyz, L, sigma, epsilon, rc):
    Avogadro = 6.02214076e23
    kJmol_to_zJ = 1.0e24 / Avogadro
    N = xyz.shape[0]
    E_total = 0.0
    for i in range(N - 1):
        ri = xyz[i]
        for j in range(i + 1, N):
            r_ij = dist(ri, xyz[j], L)
            E_total += E_ij(r_ij, sigma, epsilon, rc)
    return E_total * kJmol_to_zJ

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.3', 3)
target = targets[0]

positions1 = np.array([[1, 1, 1], [1.1, 1.1, 1.1]])
L1 = 10.0
sigma1 = 1.0
epsilon1 = 1.0
rc = 1
assert np.allclose(E_pot(positions1, L1, sigma1, epsilon1, rc), target)
target = targets[1]

positions2 = np.array([[1, 1, 1], [1, 9, 1], [9, 1, 1], [9, 9, 1]])
L2 = 10.0
sigma2 = 1.0
epsilon2 = 1.0
rc = 2
assert np.allclose(E_pot(positions2, L2, sigma2, epsilon2, rc), target)
target = targets[2]

positions3 = np.array([[3.18568952, 6.6741038,  1.31797862],
 [7.16327204, 2.89406093, 1.83191362],
 [5.86512935, 0.20107546, 8.28940029],
 [0.04695476, 6.77816537, 2.70007973],
 [7.35194022, 9.62188545, 2.48753144],
 [5.76157334, 5.92041931, 5.72251906],
 [2.23081633, 9.52749012, 4.47125379],
 [8.46408672, 6.99479275, 2.97436951],
 [8.1379782,  3.96505741, 8.81103197],
 [5.81272873, 8.81735362, 6.9253159 ]])  # 10 particles in a 10x10x10 box
L3 = 10.0
sigma3 = 1.0
epsilon3 = 1.0
rc = 5
assert np.allclose(E_pot(positions3, L3, sigma3, epsilon3, rc), target)
