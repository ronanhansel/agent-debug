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
def f_ij(r, sigma, epsilon, rc):
    """Calculate the Lennard-Jones force vector between two particles."""
    r_vec = [float(x) for x in r]
    r_mag = sum(x * x for x in r_vec) ** 0.5
    if r_mag == 0.0:
        raise ValueError("Displacement vector must be non-zero.")
    if sigma <= 0.0 or rc <= 0.0:
        raise ValueError("sigma and rc must be positive.")
    if r_mag >= rc:
        return [0.0, 0.0, 0.0]
    inv_r = sigma / r_mag
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 * inv_r6
    coef = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / (r_mag * r_mag)
    return [coef * x for x in r_vec]
def forces(N, xyz, L, sigma, epsilon, rc):
    '''Calculate the net forces acting on each particle in a system due to all pairwise interactions.
    Parameters
    ----------
    N : int
        The number of particles in the system.
    xyz : ndarray, shape (N, 3)
        Cartesian coordinates (in nm) of the particles.
    L : float
        Length of the cubic simulation box edge (nm).
    sigma : float
        Lennard-Jones size parameter (nm).
    epsilon : float
        Depth of the Lennard-Jones potential well (zJ).
    rc : float
        Cut-off distance (nm).
    Returns
    -------
    ndarray, shape (N, 3)
        Net force acting on each particle (zJ / nm).
    '''
    if xyz.shape != (N, 3):
        raise ValueError("xyz must have shape (N, 3).")
    f_xyz = np.zeros((N, 3), dtype=float)
    for i in range(N - 1):
        ri = xyz[i]
        for j in range(i + 1, N):
            r_vec = ri - xyz[j]
            r_vec -= L * np.round(r_vec / L)
            F_ij = np.asarray(f_ij(r_vec, sigma, epsilon, rc), dtype=float)
            f_xyz[i] += F_ij
            f_xyz[j] -= F_ij
    return f_xyz

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.5', 3)
target = targets[0]

N = 2
L = 10
sigma = 1
epsilon = 1
positions = np.array([[3,  -4,  5],[0.1, 0.5, 0.9]])
rc = 5
assert np.allclose(forces(N, positions, L, sigma, epsilon, rc), target)
target = targets[1]

N = 2
L = 10
sigma = 1
epsilon = 1
positions = np.array([[.62726631, 5.3077771 , 7.29719649],
       [2.25031287, 8.58926428, 4.71262908],
          [3.62726631, 1.3077771 , 2.29719649]])
rc = 7
assert np.allclose(forces(N, positions, L, sigma, epsilon, rc), target)
target = targets[2]

N = 5
L = 10
sigma = 1
epsilon = 1
positions = np.array([[.62726631, 5.3077771 , 7.29719649],
       [7.25031287, 7.58926428, 2.71262908],
       [8.7866416 , 3.73724676, 9.22676027],
       [0.89096788, 5.3872004 , 7.95350911],
       [6.068183  , 3.55807037, 2.7965242 ]])
rc = 10
assert np.allclose(forces(N, positions, L, sigma, epsilon, rc), target)
