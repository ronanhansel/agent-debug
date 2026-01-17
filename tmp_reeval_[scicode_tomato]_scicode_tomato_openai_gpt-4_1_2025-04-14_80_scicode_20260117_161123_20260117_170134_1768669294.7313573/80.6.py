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
def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
    positions = np.array(positions, dtype=float) #
    velocities = np.array(velocities, dtype=float) #
    f_xyz = np.array(forces(N, positions, L, sigma, epsilon, rc), dtype=float) #
    a_xyz = f_xyz / m
    new_positions = positions + velocities * dt + 0.5 * a_xyz * dt * dt
    new_positions = np.mod(new_positions, L) #
    f_xyz_new = np.array(forces(N, new_positions, L, sigma, epsilon, rc), dtype=float) #
    a_xyz_new = f_xyz_new / m
    new_velocities = velocities + 0.5 * (a_xyz + a_xyz_new) * dt
    return new_positions, new_velocities

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.6', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
N = 2
positions =np.array([[5.53189438, 7.24107158, 6.12400066],
 [5.48717536, 4.31571988, 6.51183]])  # N particles in a 10x10x10 box
velocities = np.array([[4.37599538, 8.91785328, 9.63675087],
 [3.83429192, 7.91712711, 5.28882593]])  # N particles in a 10x10x10 box
L = 10.0
sigma = 1.0
epsilon = 1.0
m = 1
dt = 0.01
rc = 5
assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
N = 5
positions = np.array([[4.23726683, 7.24497545, 0.05701276],
 [3.03736449, 1.48736913, 1.00346047],
 [1.9594282,  3.48694961, 4.03690693],
 [5.47580623, 4.28140578, 6.8606994 ],
 [2.04842798, 8.79815741, 0.36169018]])  # N particles in a 10x10x10 box
velocities = np.array([[6.70468142, 4.17305434, 5.5869046 ],
 [1.40388391, 1.98102942, 8.00746021],
 [9.68260045, 3.13422647, 6.92321085],
 [8.76388599, 8.9460611,  0.85043658],
 [0.39054783, 1.6983042,  8.78142503]])  # N particles in a 10x10x10 box
L = 10.0
sigma = 1.0
epsilon = 1.0
m = 1
dt = 0.01
rc = 5
assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
N = 10
positions = np.array([[4.4067278,  0.27943667, 5.56066548],
 [4.40153135, 4.25420214, 3.34203792],
 [2.12585237, 6.25071237, 3.01277888],
 [2.73834532, 6.30779077, 5.34141911],
 [1.43475136, 5.16994248, 1.90111297],
 [7.89610915, 8.58343073, 5.02002737],
 [8.51917219, 0.89182592, 5.10687864],
 [0.66107906, 4.31786204, 1.05039873],
 [1.31222278, 5.97016888, 2.28483329],
 [1.0761712,  2.30244719, 3.5953208 ]])  # N particles in a 10x10x10 box
velocities = np.array([[4.67788095, 2.01743837, 6.40407336],
 [4.83078905, 5.0524579,  3.86901721],
 [7.93680657, 5.80047382, 1.62341802],
 [7.00701382, 9.64500116, 4.99957397],
 [8.89518145, 3.41611733, 5.67142208],
 [4.27603192, 4.36804492, 7.76616414],
 [5.35546945, 9.53684998, 5.44150931],
 [0.8219327,  3.66440749, 8.50948852],
 [4.06178341, 0.27105663, 2.47080537],
 [0.67142725, 9.93850366, 9.70578668]])  # N particles in a 10x10x10 box
L = 10.0
sigma = 1.0
epsilon = 1.0
m = 1
dt = 0.01
rc = 5
assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)
