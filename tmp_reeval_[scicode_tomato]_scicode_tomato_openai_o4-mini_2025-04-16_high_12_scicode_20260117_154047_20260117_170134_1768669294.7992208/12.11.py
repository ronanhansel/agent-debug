# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    hbar = 1.054571817e-34
    m_e = 9.10938356e-31
    e_c = 1.602176634e-19
    epsilon0 = 8.8541878128e-12
    Z = 1
    V_coeff = -Z * e_c**2 / (4 * np.pi * epsilon0)
    f_r = (l * (l + 1) / r_grid**2) + (2 * m_e / hbar**2) * (V_coeff / r_grid - energy)
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    '''Given precomputed function f(r), solve the differential equation u''(r) = f(r)*u(r)
    using the Numerov method.
    Inputs:
    - f_in: input function f(r); a 1D array of float representing the function values at discretized points.
    - u_at_0: the value of u at r = 0; a float.
    - up_at_0: the derivative of u at r = 0; a float.
    - step: step size; a float.
    Output:
    - u: the integration results at each point in the radial grid; a 1D array of float.
    '''
    N = len(f_in)
    u = np.zeros(N, dtype=float)
    u[0] = u_at_0
    if N > 1:
        u[1] = u_at_0 + step * up_at_0 + 0.5 * step**2 * f_in[0] * u_at_0
        for n in range(1, N - 1):
            f_nm1 = f_in[n - 1]
            f_n   = f_in[n]
            f_np1 = f_in[n + 1]
            a = 1.0 - (5.0 * step**2 * f_n) / 12.0
            b = 1.0 + (       step**2 * f_nm1) / 12.0
            c = 1.0 + (       step**2 * f_np1) / 12.0
            u[n + 1] = (2.0 * a * u[n] - b * u[n - 1]) / c
    return u
def compute_Schrod(energy, r_grid, l):
    """
    Solve the radial Schrödinger equation u''(r) = f(r) u(r) for a given energy
    and angular momentum l using the Numerov method, and normalize the result.

    Inputs:
    - energy: float, trial energy (in SI units, Joules)
    - r_grid: 1D numpy array of floats, radial grid points (assumed ascending)
    - l: int, angular momentum quantum number

    Output:
    - ur_norm: 1D numpy array of floats, normalized radial wavefunction u(r)
    """
    f_r = f_Schrod(energy, l, r_grid)
    r_rev = r_grid[::-1]
    f_rev = f_r[::-1]
    step = r_rev[0] - r_rev[1]
    u0 = 0.0
    up0 = -1e-7
    u_rev = Numerov(f_rev, u0, up0, step)
    u = u_rev[::-1]
    norm = np.sqrt(integrate.simps(u * u, r_grid))
    ur_norm = u / norm
    return ur_norm
def shoot(energy, r_grid, l):
    """
    Input:
        energy : float
        r_grid : 1D numpy array of floats (ascending radial points, r>0)
        l      : int, angular momentum quantum number
    Output:
        f_at_0 : float, extrapolated value at r=0 of u(r)/r^l
    """
    u = compute_Schrod(energy, r_grid, l)
    v = u / (r_grid ** l)
    r0, r1 = r_grid[0], r_grid[1]
    v0, v1 = v[0], v[1]
    slope = (v1 - v0) / (r1 - r0)
    f_at_0 = v0 - slope * r0
    return f_at_0
def find_bound_states(r_grid, l, energy_grid):
    """
    Search for bound-state energies for angular momentum quantum number l
    by finding zeros of shoot(E, r_grid, l) in the given energy grid.

    Inputs:
    - r_grid: 1D numpy array of floats (ascending radial points, r > 0)
    - l: int, angular momentum quantum number
    - energy_grid: 1D array of floats, trial energies in ascending order

    Output:
    - bound_states: list of tuples (l, E) for each bound state found (up to 10)
    """
    bound_states = []
    max_states = 10
    f_prev = shoot(energy_grid[0], r_grid, l)
    for E_left, E_right in zip(energy_grid[:-1], energy_grid[1:]):
        f_right = shoot(E_right, r_grid, l)
        if f_prev * f_right < 0:
            try:
                E_root = optimize.brentq(
                    lambda E: shoot(E, r_grid, l),
                    E_left, E_right
                )
                bound_states.append((l, E_root))
                if len(bound_states) >= max_states:
                    break
            except ValueError:
                pass
        f_prev = f_right
    return bound_states
def sort_states(bound_states):
    sorted_states = sorted(
        bound_states,
        key=lambda state: state[1] + state[0] / 10000.0
    )
    return sorted_states
def calculate_charge_density(bound_states, r_grid, Z):
    """
    Input:
        bound_states: list of tuples (l, E) returned by find_bound_states
        r_grid: 1D numpy array of floats (ascending radii)
        Z: int, total number of electrons (atomic number)
    Output:
        charge_density: 1D numpy array of floats, the radial charge density
    """
    sorted_states = sort_states(bound_states)
    charge_density = np.zeros_like(r_grid, dtype=float)
    Z_remaining = Z
    for l, E in sorted_states:
        if Z_remaining <= 0:
            break
        degeneracy = 2 * (2 * l + 1)
        occupancy = min(degeneracy, Z_remaining)
        u_r = compute_Schrod(E, r_grid, l)
        charge_density += occupancy * (u_r ** 2)
        Z_remaining -= occupancy
    return charge_density
def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    """
    Solve the radial Poisson equation ∇²V_H(r) = -8π ρ(r) for the Hartree potential
    using an inhomogeneous Numerov scheme on U(r) = r·V_H(r).

    Inputs:
      charge_density : 1D array of floats, ρ(r) at each grid point
      u_at_0         : float, U(0) initial value
      up_at_0        : float, U'(0) initial derivative
      step           : float, radial grid spacing h
      r_grid         : 1D array of ascending radial points
      Z              : int, atomic number (unused here)
    Output:
      U              : 1D array of floats, the HartreeU term U(r)=r·V_H(r)
    """
    N = len(r_grid)
    S = -8.0 * np.pi * r_grid * charge_density
    U = np.zeros_like(r_grid, dtype=float)
    U[0] = u_at_0
    if N > 1:
        U[1] = u_at_0 + step * up_at_0 + 0.5 * step**2 * S[0]
        for n in range(1, N - 1):
            U[n + 1] = (
                2.0 * U[n]
                - U[n - 1]
                + (step**2 / 12.0) * (S[n + 1] + 10.0 * S[n] + S[n - 1])
            )
    return U
def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy   : float
    r_grid   : 1D numpy array of radii
    l        : int, angular momentum quantum number
    Z        : int, atomic number
    hartreeU : 1D numpy array, U(r)=r·V_H(r)
    Output
    f_r      : 1D numpy array, the function values for u''(r)=f(r)u(r)
    '''
    hbar = 1.054571817e-34
    m_e = 9.10938356e-31
    e_c = 1.602176634e-19
    epsilon0 = 8.8541878128e-12
    V_coeff = -Z * e_c**2 / (4.0 * np.pi * epsilon0)
    V_H = hartreeU / r_grid
    V_total = V_coeff / r_grid + V_H
    f_r = (l * (l + 1) / r_grid**2) + (2.0 * m_e / hbar**2) * (V_total - energy)
    return f_r
def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    """
    Solve the radial Schrödinger equation with the Hartree potential term
    u''(r) = f_Schrod_Hartree(r) * u(r) using the Numerov method, and normalize.

    Inputs:
    - energy   : float, trial energy (SI units, Joules)
    - r_grid   : 1D numpy array of floats, ascending radial grid points (r>0)
    - l        : int, angular momentum quantum number
    - Z        : int, atomic number
    - hartreeU : 1D numpy array of floats, U(r)=r·V_H(r) from the Poisson solver

    Output:
    - ur_norm  : 1D numpy array of floats, normalized radial wavefunction u(r)
    """
    f_r = f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    r_rev = r_grid[::-1]
    f_rev = f_r[::-1]
    step = r_rev[0] - r_rev[1]
    u0, up0 = 0.0, -1e-7
    u_rev = Numerov(f_rev, u0, up0, step)
    u = u_rev[::-1]
    norm = np.sqrt(integrate.simpson(u * u, r_grid))
    ur_norm = u / norm
    return ur_norm
def extrapolate_polyfit(energy, r_grid, l, Z, hartreeU):
    """
    Extrapolate u(r) at r=0 by fitting u(r)/r^l versus r to a 3rd-degree polynomial
    using the first four grid points, then evaluating at r=0.

    Inputs:
    - energy   : float, trial energy (SI units, Joules)
    - r_grid   : 1D numpy array of floats, ascending radial grid points (r > 0)
    - l        : int, angular momentum quantum number
    - Z        : int, atomic number
    - hartreeU : 1D numpy array of floats, U(r)=r·V_H(r) from the Poisson solver

    Output:
    - u0       : float, extrapolated value of u(r) at r = 0
    """
    u = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    r_small = r_grid[:4]
    u_small = u[:4]
    v_small = u_small / (r_small**l)
    coeffs = np.polyfit(r_small, v_small, 3)
    v0 = np.polyval(coeffs, 0.0)
    u0 = v0 if l == 0 else 0.0
    return u0

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.11', 3)
target = targets[0]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=28
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(extrapolate_polyfit(-0.5, r_grid, 2, Z, hu), target)
target = targets[1]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 3
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(extrapolate_polyfit(-0.4, r_grid, 2, Z, hu), target)
target = targets[2]

energy_grid = -0.9/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(extrapolate_polyfit(-0.5, r_grid, 3, Z, hu), target)
