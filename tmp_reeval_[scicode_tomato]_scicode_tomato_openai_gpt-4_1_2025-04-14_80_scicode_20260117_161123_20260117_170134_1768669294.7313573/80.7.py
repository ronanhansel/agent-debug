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
def MD_NVT(positions, velocities, masses, forces, dt, n_steps, temperature, collision_frequency, boltzmann_constant, rng, get_forces_func, periodic_box=None):
    n_particles = len(masses)
    state_history = []
    total_energy = None
    for step in range(n_steps):
        velocities = velocities + 0.5 * dt * (forces / masses[:, None])
        positions = positions + velocities * dt
        if periodic_box is not None:
            positions = positions % periodic_box
        for i in range(n_particles):
            if rng.uniform(0, 1) < 1 - pow(2.718281828459045, -collision_frequency * dt):
                stddev = (boltzmann_constant * temperature / masses[i]) ** 0.5
                velocities[i] = rng.normal(0, stddev, size=velocities.shape[1])
        forces = get_forces_func(positions)
        velocities = velocities + 0.5 * dt * (forces / masses[:, None])
        kinetic_energy = 0.5 * (masses[:, None] * velocities**2).sum()
        temperature_inst = (2.0 / (velocities.size)) * kinetic_energy / boltzmann_constant
        total_energy = kinetic_energy
        state = {
            'step': step,
            'positions': positions.copy(),
            'velocities': velocities.copy(),
            'forces': forces.copy(),
            'temperature': temperature_inst,
            'kinetic_energy': kinetic_energy,
            'total_energy': total_energy
        }
        state_history.append(state)
    return state_history

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
