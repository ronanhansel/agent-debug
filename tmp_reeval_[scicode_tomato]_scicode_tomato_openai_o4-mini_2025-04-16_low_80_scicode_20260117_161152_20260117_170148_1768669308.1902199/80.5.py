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
def E_ij(r, sigma, epsilon, rc):
    '''Calculate the truncated-and-shifted Lennard-Jones potential between two particles.

    Parameters:
        r (float): Distance between particles i and j.
        sigma (float): Zero‐crossing distance of the Lennard-Jones potential.
        epsilon (float): Depth of the Lennard-Jones well.
        rc (float): Cutoff distance beyond which the potential is set to zero.

    Returns:
        float: Truncated and shifted Lennard-Jones potential energy.
    '''
    if r >= rc:
        return 0.0
    inv_r6 = (sigma / r) ** 6
    V_r = 4.0 * epsilon * (inv_r6 * inv_r6 - inv_r6)
    inv_rc6 = (sigma / rc) ** 6
    V_rc = 4.0 * epsilon * (inv_rc6 * inv_rc6 - inv_rc6)
    return V_r - V_rc
def E_pot(xyz, L, sigma, epsilon, rc):
    N = xyz.shape[0]
    E_tot = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            r_ij = dist(xyz[i], xyz[j], L)
            E_tot += E_ij(r_ij, sigma, epsilon, rc)
    conversion_factor = 1e24 / Avogadro
    return E_tot * conversion_factor
def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles using the truncated-and-shifted
    Lennard-Jones potential.

    Parameters:
        r : array_like
            The 3D displacement vector from particle j to particle i (shape: (3,)).
        sigma (float): The distance at which the Lennard-Jones potential crosses zero.
        epsilon (float): The depth of the potential well.
        rc (float): The cutoff distance beyond which the force is zero.

    Returns:
        list of float: The force vector on particle i due to particle j.
    '''
    r0, r1, r2 = r
    r_norm = (r0*r0 + r1*r1 + r2*r2)**0.5
    if r_norm >= rc or r_norm == 0.0:
        return [0.0, 0.0, 0.0]
    inv_r7 = (sigma**6) / (r_norm**7)
    inv_r13 = (sigma**12) / (r_norm**13)
    force_mag = 4.0 * epsilon * (12.0 * inv_r13 - 6.0 * inv_r7)
    return [
        force_mag * (r0 / r_norm),
        force_mag * (r1 / r_norm),
        force_mag * (r2 / r_norm)
    ]
def forces(N, xyz, L, sigma, epsilon, rc):
    '''Calculate the net forces acting on each particle in a system due to all pairwise interactions.'''
    f_xyz = np.zeros((N, 3))
    for i in range(N - 1):
        for j in range(i + 1, N):
            rij = xyz[i] - xyz[j]
            rij -= L * np.round(rij / L)
            fij = f_ij(rij, sigma, epsilon, rc)
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
