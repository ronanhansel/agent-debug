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
def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
    '''Integrate one time-step using the Velocity Verlet algorithm for a system
    of N particles interacting via a Lennard-Jones potential.

    Parameters
    ----------
    N : int
        Number of particles.
    sigma : float
        Lennard-Jones σ parameter (nm).
    epsilon : float
        Lennard-Jones well depth (zJ).
    positions : ndarray, shape (N, 3)
        Current particle coordinates (nm).
    velocities : ndarray, shape (N, 3)
        Current particle velocities (nm ps⁻¹).
    dt : float
        Time-step (ps).
    m : float
        Particle mass (zJ ps² nm⁻²).
    L : float
        Cubic box edge length (nm).
    rc : float
        Lennard-Jones cut-off distance (nm).

    Returns
    -------
    new_positions : ndarray, shape (N, 3)
        Updated coordinates after one time-step.
    new_velocities : ndarray, shape (N, 3)
        Updated velocities after one time-step.
    '''
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.shape != (N, 3) or velocities.shape != (N, 3):
        raise ValueError("`positions` and `velocities` must both have shape (N, 3).")
    if dt <= 0.0:
        raise ValueError("`dt` must be positive.")
    if m <= 0.0:
        raise ValueError("`m` must be positive.")
    F_old = forces(N, positions, L, sigma, epsilon, rc)
    a_old = F_old / m
    new_positions = positions + velocities * dt + 0.5 * a_old * dt * dt
    new_positions %= L
    F_new = forces(N, new_positions, L, sigma, epsilon, rc)
    a_new = F_new / m
    new_velocities = velocities + 0.5 * (a_old + a_new) * dt
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
