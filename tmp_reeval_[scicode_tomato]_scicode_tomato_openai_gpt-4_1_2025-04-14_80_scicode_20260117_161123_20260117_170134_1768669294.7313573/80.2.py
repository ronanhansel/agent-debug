import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro

def dist(r1, r2, L):
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    distance = (dx*dx + dy*dy + dz*dz) ** 0.5
    return distance
def E_ij(r, sigma, epsilon, rc):
    if r >= rc:
        return 0.0
    sr6 = (sigma / r) ** 6
    V = 4 * epsilon * (sr6 ** 2 - sr6)
    src6 = (sigma / rc) ** 6
    Vc = 4 * epsilon * (src6 ** 2 - src6)
    E = V - Vc
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
