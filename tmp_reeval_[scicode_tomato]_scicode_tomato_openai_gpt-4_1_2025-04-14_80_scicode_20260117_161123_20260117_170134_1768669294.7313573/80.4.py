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
def E_pot(xyz, L, sigma, epsilon, rc):
    N = xyz.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = dist(xyz[i], xyz[j], L)
            e_ij = E_ij(r_ij, sigma, epsilon, rc)
            E += e_ij
    return E
def f_ij(r, sigma, epsilon, rc):
    r = np.array(r, dtype=float) #
    norm = np.linalg.norm(r) #
    if norm >= rc or norm == 0.0:
        return np.zeros(3) #
    sr = sigma / norm
    sr6 = sr ** 6
    force_scalar = 24 * epsilon / norm * (2 * sr6 * sr6 - sr6)
    return force_scalar * (r / norm)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.4', 3)
target = targets[0]

sigma = 1
epsilon = 1
r = np.array([-3.22883506e-03,  2.57056485e+00,  1.40822287e-04])
rc = 1
assert np.allclose(f_ij(r,sigma,epsilon,rc), target)
target = targets[1]

sigma = 2
epsilon = 3
r = np.array([3,  -4,  5])
rc = 10
assert np.allclose(f_ij(r,sigma,epsilon,rc), target)
target = targets[2]

sigma = 3
epsilon = 7
r = np.array([5,  9,  7])
rc = 20
assert np.allclose(f_ij(r,sigma,epsilon,rc), target)
