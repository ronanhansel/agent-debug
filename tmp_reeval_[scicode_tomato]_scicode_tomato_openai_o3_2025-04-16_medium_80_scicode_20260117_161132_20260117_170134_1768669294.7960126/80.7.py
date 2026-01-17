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
def MD_NVT(
    N,
    init_positions,
    init_velocities,
    L,
    sigma,
    epsilon,
    rc,
    m,
    dt,
    num_steps,
    T,
    nu,
):
    '''Integrate the equations of motion with Velocity Verlet while applying an
    Anderson thermostat for temperature control (NVT ensemble).

    Returns
    -------
    tuple
        E_total_array        : ndarray (num_steps,)  – total energy (zJ) each step
        instant_T_array      : ndarray (num_steps,) – instantaneous temperature (K)
        positions            : ndarray (N, 3)       – final positions (nm)
        velocities           : ndarray (N, 3)       – final velocities (nm ps⁻¹)
        intercollision_times : ndarray              – ps between stochastic collisions
    '''
    k_B_zJ = 1.380649e-23 / 1.0e-21
    m_kg_per_particle = (m * 1.0e-3) / Avogadro
    m_sim = m_kg_per_particle / 1.0e-27
    if m_sim <= 0.0:
        raise ValueError("Particle mass must be positive.")
    sigma_v = math.sqrt(k_B_zJ * T / m_sim)
    p_collision = 1.0 - math.exp(-nu * dt)
    E_total_array = np.zeros(num_steps, dtype=float)
    instant_T_array = np.zeros(num_steps, dtype=float)
    intercollision_times = []
    time_since_last_collision = 0.0
    positions = np.asarray(init_positions, dtype=float).copy()
    velocities = np.asarray(init_velocities, dtype=float).copy()
    if positions.shape != (N, 3) or velocities.shape != (N, 3):
        raise ValueError("`init_positions` and `init_velocities` must each be (N, 3).")
    for step in range(num_steps):
        positions, velocities = velocity_verlet(
            N, sigma, epsilon, positions, velocities, dt, m_sim, L, rc
        )
        rand = np.random.rand(N)
        collide_mask = rand < p_collision
        n_collide = collide_mask.sum()
        if n_collide:
            velocities[collide_mask] = np.random.normal(
                loc=0.0, scale=sigma_v, size=(n_collide, 3)
            )
        time_since_last_collision += dt
        if n_collide:
            intercollision_times.append(time_since_last_collision)
            time_since_last_collision = 0.0
        KE = 0.5 * m_sim * np.sum(velocities ** 2)
        PE = E_pot(positions, L, sigma, epsilon, rc)
        E_total_array[step] = KE + PE
        instant_T_array[step] = (2.0 * KE) / (3.0 * N * k_B_zJ)
    return (
        E_total_array,
        instant_T_array,
        positions,
        velocities,
        np.asarray(intercollision_times, dtype=float),
    )

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
