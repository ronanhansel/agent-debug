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
