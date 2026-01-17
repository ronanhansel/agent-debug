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
def E_pot(xyz, L, sigma, epsilon, rc):
    N = len(xyz)
    E = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            r = dist(xyz[i], xyz[j], L)
            E += E_ij(r, sigma, epsilon, rc)
    return E
def f_ij(r, sigma, epsilon, rc):
    dx, dy, dz = r
    r_mag = (dx*dx + dy*dy + dz*dz)**0.5
    if r_mag >= rc or r_mag == 0.0:
        return [0.0, 0.0, 0.0]
    force_scalar = 24.0 * epsilon * (2.0 * sigma**12 / r_mag**13 - sigma**6 / r_mag**7)
    return [
        force_scalar * dx / r_mag,
        force_scalar * dy / r_mag,
        force_scalar * dz / r_mag
    ]
def forces(N, xyz, L, sigma, epsilon, rc):
    f_xyz = np.zeros((N, 3), dtype=float)
    for i in range(N - 1):
        for j in range(i + 1, N):
            delta = xyz[i] - xyz[j]
            delta -= L * np.round(delta / L)
            fij = np.array(f_ij(delta, sigma, epsilon, rc), dtype=float)
            f_xyz[i] += fij
            f_xyz[j] -= fij
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
