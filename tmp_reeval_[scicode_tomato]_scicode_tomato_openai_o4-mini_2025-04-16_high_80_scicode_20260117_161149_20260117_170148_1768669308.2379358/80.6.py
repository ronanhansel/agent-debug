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
def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
    """
    Run one timestep of the Velocity–Verlet integration for N particles
    interacting via a truncated–shifted Lennard–Jones potential.

    Parameters:
        N (int): Number of particles.
        sigma (float): Lennard–Jones zero-crossing distance.
        epsilon (float): Depth of the potential well.
        positions (ndarray): Shape (N,3), current particle positions.
        velocities (ndarray): Shape (N,3), current particle velocities.
        dt (float): Time step.
        m (float): Particle mass.
        L (float): Side length of the cubic periodic box.
        rc (float): Cutoff distance for the Lennard–Jones potential.

    Returns:
        new_positions (ndarray): Shape (N,3), updated positions.
        new_velocities (ndarray): Shape (N,3), updated velocities.
    """
    F_current = forces(N, positions, L, sigma, epsilon, rc)
    new_positions = positions + velocities * dt + (F_current / m) * (0.5 * dt**2)
    new_positions = np.mod(new_positions, L)
    F_new = forces(N, new_positions, L, sigma, epsilon, rc)
    new_velocities = velocities + ((F_current + F_new) / (2.0 * m)) * dt
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
