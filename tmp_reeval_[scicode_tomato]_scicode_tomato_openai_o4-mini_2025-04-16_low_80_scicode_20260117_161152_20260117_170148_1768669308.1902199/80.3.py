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
