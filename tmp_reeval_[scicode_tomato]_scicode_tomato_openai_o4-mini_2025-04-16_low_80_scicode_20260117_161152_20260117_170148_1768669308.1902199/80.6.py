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
def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
    """
    This function runs the Velocity Verlet algorithm to integrate the positions and velocities
    of atoms interacting through a Lennard-Jones potential forward by one time step.

    Parameters:
        N (int): Number of particles in the system.
        sigma (float): LJ zero‐crossing distance.
        epsilon (float): LJ well depth.
        positions (ndarray[N,3]): Current positions of all atoms.
        velocities (ndarray[N,3]): Current velocities of all atoms.
        dt (float): Time step size.
        m (float): Particle mass.
        L (float): Length of the cubic box (periodic boundary).
        rc (float): Cutoff distance for LJ interactions.

    Returns:
        new_positions (ndarray[N,3]): Updated positions after one time step.
        new_velocities (ndarray[N,3]): Updated velocities after one time step.
    """
    F_old = forces(N, positions, L, sigma, epsilon, rc)
    a_old = F_old / m
    new_positions = positions + velocities * dt + 0.5 * a_old * dt**2
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
