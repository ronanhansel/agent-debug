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
def MD_NVT(N, init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu):
    positions = init_positions.copy()
    velocities = init_velocities.copy()
    E_total_array = np.zeros(num_steps)
    instant_T_array = np.zeros(num_steps)
    intercollision_times = []
    m_particle = (m / 1000.0) / Avogadro
    v_std_nmps = np.sqrt(k * T / m_particle) * 1e-3
    p_coll = 1.0 - np.exp(-nu * dt)
    for step in range(num_steps):
        positions, velocities = velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc)
        for i in range(N):
            if np.random.rand() < p_coll:
                t_ic = -np.log(np.random.rand()) / nu
                intercollision_times.append(t_ic)
                velocities[i] = np.random.normal(0.0, v_std_nmps, size=3)
        v2_sum_si = np.sum((velocities * 1e3) ** 2)
        T_inst = (m_particle * v2_sum_si) / (3.0 * N * k)
        instant_T_array[step] = T_inst
        K_J = 0.5 * m_particle * v2_sum_si
        K_zJ = K_J * 1e21
        E_pot_zJ = E_pot(positions, L, sigma, epsilon, rc)
        E_total_array[step] = K_zJ + E_pot_zJ
    return E_total_array, instant_T_array, positions, velocities, intercollision_times

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.7', 3)
target = targets[0]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return [np.array(position),L]
m = 1
sigma = 1
epsilon = 1
dt = 0.001
nu = 0.2/dt
T = 1.5
N = 200
rc = 2.5
num_steps = 2000
init_positions, L = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
E_total_array,instant_T_array,positions,velocities,intercollision_times = MD_NVT(N,init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu)
T_thresh=0.05
s1 = np.abs(np.mean(instant_T_array[int(num_steps*0.2):])-T)/T < T_thresh
nu_thresh=0.05
v1 = np.abs(np.mean(1/np.mean(intercollision_times))-nu)/nu < nu_thresh
assert (s1, v1) == target
target = targets[1]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return [np.array(position),L]
m = 1
sigma = 1
epsilon = 1
dt = 0.0001
nu = 0.2/dt
T = 100
N = 200
rc = 2.5
num_steps = 2000
init_positions, L = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
E_total_array,instant_T_array,positions,velocities,intercollision_times = MD_NVT(N,init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu)
T_thresh=0.05
s1 = np.abs(np.mean(instant_T_array[int(num_steps*0.2):])-T)/T < T_thresh
nu_thresh=0.05
v1 = np.abs(np.mean(1/np.mean(intercollision_times))-nu)/nu < nu_thresh
assert (s1, v1) == target
target = targets[2]

import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return [np.array(position),L]
m = 1
sigma = 1
epsilon = 1
dt = 0.001
nu = 0.2/dt
T = 0.01
N = 200
rc = 2
num_steps = 2000
init_positions, L = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
E_total_array,instant_T_array,positions,velocities,intercollision_times = MD_NVT(N,init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu)
T_thresh=0.05
s1 = np.abs(np.mean(instant_T_array[int(num_steps*0.2):])-T)/T < T_thresh
nu_thresh=0.05
v1 = np.abs(np.mean(1/np.mean(intercollision_times))-nu)/nu < nu_thresh
assert (s1, v1) == target
