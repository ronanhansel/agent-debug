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
def MD_NVT(N, init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu):
    """
    Integrate the equations of motion using the velocity Verlet algorithm,
    with inclusion of the Anderson thermostat for temperature control.

    Returns:
        E_total_array: (num_steps,) array of total energies (zJ)
        instant_T_array: (num_steps,) array of instantaneous temperatures (K)
        positions: (N,3) array of final positions (nm)
        velocities: (N,3) array of final velocities (nm/ps)
        intercollision_times: list of time intervals (ps) between stochastic collisions
    """
    E_total_array   = np.zeros(num_steps, dtype=float)
    instant_T_array = np.zeros(num_steps, dtype=float)
    positions  = init_positions.copy()
    velocities = init_velocities.copy()
    mass_particle_kg = (m * 1e-3) / Avogadro
    k_B = sp.constants.k
    sigma_v_si  = math.sqrt(k_B * T / mass_particle_kg)
    sigma_v_sim = sigma_v_si / 1e3
    elapsed_since_last_collision = 0.0
    intercollision_times = []
    for step in range(num_steps):
        positions, velocities = velocity_verlet(
            N, sigma, epsilon, positions, velocities, dt, m, L, rc
        )
        elapsed_since_last_collision += dt
        p_coll = nu * dt
        for i in range(N):
            if np.random.rand() < p_coll:
                intercollision_times.append(elapsed_since_last_collision)
                elapsed_since_last_collision = 0.0
                velocities[i, :] = np.random.normal(0.0, sigma_v_sim, size=3)
        v_si = velocities * 1e3
        ke_J  = 0.5 * mass_particle_kg * np.sum(v_si**2)
        ke_zJ = ke_J * 1e21
        pe_zJ = E_pot(positions, L, sigma, epsilon, rc)
        E_total_array[step]   = ke_zJ + pe_zJ
        instant_T_array[step] = (2.0 * ke_J) / (3.0 * N * k_B)
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
