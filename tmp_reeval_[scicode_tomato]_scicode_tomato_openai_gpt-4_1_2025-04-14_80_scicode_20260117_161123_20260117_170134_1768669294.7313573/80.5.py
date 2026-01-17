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
def forces(N, xyz, L, sigma, epsilon, rc):
    def minimum_image_displacement(r1, r2, L):
        dr = r1 - r2
        dr -= L * round(dr / L) # np.round should be replaced, but the logic should be preserved
        return dr

    def f_ij(r, sigma, epsilon, rc):
        norm = (r[0]**2 + r[1]**2 + r[2]**2) ** 0.5
        if norm >= rc or norm == 0.0:
            return [0., 0., 0.]
        sr = sigma / norm
        sr6 = sr ** 6
        force_scalar = 24 * epsilon / norm * (2 * sr6 * sr6 - sr6)
        return [force_scalar * (r[0] / norm), force_scalar * (r[1] / norm), force_scalar * (r[2] / norm)]

    f_xyz = []
    for i in range(N):
        f_xyz.append([0., 0., 0.])
    for i in range(N):
        for j in range(i + 1, N):
            disp = [xyz[i][k] - xyz[j][k] for k in range(3)]
            for k in range(3):
                disp[k] -= L * round(disp[k] / L)
            fij = f_ij(disp, sigma, epsilon, rc)
            for k in range(3):
                f_xyz[i][k] += fij[k]
                f_xyz[j][k] -= fij[k]
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
