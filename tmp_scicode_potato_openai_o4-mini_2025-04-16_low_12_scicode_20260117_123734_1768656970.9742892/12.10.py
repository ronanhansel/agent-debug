from scipy import integrate
from scipy import optimize
import numpy as np
# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson


def f_Schrod(energy, l, r_grid):
    hbar = 1.054571817e-34
    m_e = 9.10938356e-31
    e_charge = 1.602176634e-19
    epsilon0 = 8.8541878128e-12
    term_cent = l * (l + 1) / r_grid**2
    V_coul = - e_charge**2 / (4 * np.pi * epsilon0 * r_grid)
    f_r = term_cent + (2 * m_e / hbar**2) * (V_coul - energy)
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    N = len(f_in)
    u = [0.0] * N
    if N == 0:
        return u
    u[0] = u_at_0
    if N > 1:
        u[1] = u_at_0 + step * up_at_0 + 0.5 * step * step * f_in[0] * u_at_0
    h2 = step * step
    for i in range(1, N - 1):
        denom = 1.0 - (h2 / 12.0) * f_in[i + 1]
        term_i = 2.0 * u[i] * (1.0 + (5.0 * h2 / 12.0) * f_in[i])
        term_im1 = u[i - 1] * (1.0 - (h2 / 12.0) * f_in[i - 1])
        u[i + 1] = (term_i - term_im1) / denom
    return u
def compute_Schrod(energy, r_grid, l):
    """
    Solve the radial Schrödinger equation u''(r) = f(r) u(r)
    using f_Schrod and Numerov, then normalize u(r) with Simpson's rule.

    Input
    -----
    energy : float
        The eigenenergy E (in joules).
    r_grid : array_like
        The radial grid points (1D ascending array).
    l : int
        Angular momentum quantum number.

    Output
    ------
    ur_norm : ndarray
        The normalized radial wavefunction u(r) on r_grid.
    """
    r = np.asarray(r_grid)
    r_rev = r[::-1]
    h = r[0] - r[1]
    f_rev = f_Schrod(energy, l, r_rev)
    u_rev = Numerov(f_rev, u_at_0=0.0, up_at_0=-1e-7, step=h)
    u = np.asarray(u_rev)[::-1]
    norm = integrate.simps(u**2, r)
    ur_norm = u / np.sqrt(norm)
    return ur_norm
def shoot(energy, r_grid, l):
    """
    Input 
    -----
    energy : float
        Eigenenergy E in joules.
    r_grid : array_like
        1D ascending radial grid points.
    l : int
        Angular momentum quantum number.

    Output
    ------
    f_at_0 : float
        The linearly extrapolated value at r = 0 of [u(r)/r^l],
        where u(r) is the normalized radial solution from compute_Schrod.
    """
    u = compute_Schrod(energy, r_grid, l)
    r0, r1 = r_grid[0], r_grid[1]
    f0 = u[0] / (r0**l)
    f1 = u[1] / (r1**l)
    slope = (f1 - f0) / (r1 - r0)
    f_at_0 = f0 - slope * r0
    return f_at_0
def find_bound_states(r_grid, l, energy_grid):
    """
    Input
    -----
    r_grid : array_like
        1D ascending radial grid points.
    l : int
        Angular momentum quantum number.
    energy_grid : array_like
        1D array of trial energies (in joules).

    Output
    ------
    bound_states : list of tuples
        Each tuple is (l, E_root) for a found bound-state energy E_root.
    """
    bound_states = []
    max_states = 10

    energies = np.asarray(energy_grid, dtype=float)
    if energies.size < 2:
        return bound_states
    if not np.all(np.diff(energies) >= 0):
        energies = np.sort(energies)

    for i in range(len(energies) - 1):
        E_low, E_high = energies[i], energies[i + 1]
        f_low = shoot(E_low, r_grid, l)
        f_high = shoot(E_high, r_grid, l)

        if f_low * f_high < 0.0:
            try:
                E_root = optimize.brentq(
                    lambda E: shoot(E, r_grid, l),
                    E_low, E_high
                )
                bound_states.append((l, E_root))
                if len(bound_states) >= max_states:
                    break
            except ValueError:
                pass

    return bound_states
def sort_states(bound_states):
    return sorted(bound_states, key=lambda state: state[1] + state[0] / 10000.0)
def calculate_charge_density(bound_states, r_grid, Z):
    """
    Calculate the radius-dependent electronic charge density from bound states.

    Parameters
    ----------
    bound_states : list of (l, energy) tuples
        Bound states as returned by find_bound_states.
    r_grid : array_like
        1D ascending radial grid.
    Z : int
        Total number of electrons to place.

    Returns
    -------
    charge_density : ndarray
        Radial charge density ρ(r) on r_grid.
    """
    sorted_states = sort_states(bound_states)
    electrons_left = Z
    charge_density = np.zeros_like(r_grid, dtype=float)
    for l, energy in sorted_states:
        if electrons_left <= 0:
            break
        degeneracy = 2 * (2 * l + 1)
        occupancy = min(degeneracy, electrons_left)
        u_r = compute_Schrod(energy, r_grid, l)
        psi_r = u_r / r_grid
        charge_density += occupancy * (psi_r ** 2)
        electrons_left -= occupancy
    return charge_density
def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    r = [float(x) for x in r_grid]
    N = len(r)
    U = [0.0] * N
    S = [-8.0 * 3.141592653589793 * r[i] * float(charge_density[i]) for i in range(N)]
    if N == 0:
        return U
    U[0] = u_at_0
    if N > 1:
        U[1] = u_at_0 + up_at_0 * step + 0.5 * S[0] * step**2
    for i in range(1, N - 1):
        U[i + 1] = 2.0 * U[i] - U[i - 1] + step**2 * S[i]
    return U
def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    """
    Input 
    -----
    energy : float
        Eigenenergy E (in joules).
    r_grid : array_like
        1D ascending radial grid points.
    l : int
        Angular momentum quantum number.
    Z : int
        Nuclear charge.
    hartreeU : array_like
        Hartree potential term U(r) = V_H(r) * r on the same grid.

    Output
    ------
    f_r : ndarray
        The array f(r) for the radial Schrödinger equation u''(r) = f(r) u(r)
        including the Hartree potential.
    """
    hbar = 1.054571817e-34
    m_e = 9.10938356e-31
    e_charge = 1.602176634e-19
    epsilon0 = 8.8541878128e-12
    r = np.asarray(r_grid, dtype=float)
    U = np.asarray(hartreeU, dtype=float)
    term_cent = l * (l + 1) / (r**2)
    V_coul = -Z * e_charge**2 / (4.0 * np.pi * epsilon0 * r)
    V_H = U / r
    f_r = term_cent + (2.0 * m_e / hbar**2) * (V_coul + V_H - energy)
    return f_r
def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    """
    Solve the radial Schrödinger equation u''(r) = f(r) u(r)
    including the Hartree potential term, using Numerov integration
    and normalize the result with Simpson's rule.

    Parameters
    ----------
    energy : float
        Eigenenergy E (in joules).
    r_grid : array_like
        1D ascending radial grid points.
    l : int
        Angular momentum quantum number.
    Z : int
        Atomic (nuclear) charge.
    hartreeU : array_like
        Hartree term U(r) = V_H(r) * r on the same ascending grid.

    Returns
    -------
    ur_norm : ndarray
        The normalized radial wavefunction u(r) on r_grid.
    """
    r = np.asarray(r_grid, dtype=float)
    U = np.asarray(hartreeU, dtype=float)
    h = r[0] - r[1]
    r_rev = r[::-1]
    U_rev = U[::-1]
    f_rev = f_Schrod_Hartree(energy, r_rev, l, Z, U_rev)
    u_rev = Numerov(f_rev, u_at_0=0.0, up_at_0=-1e-7, step=h)
    u = np.asarray(u_rev)[::-1]
    norm = integrate.simps(u**2, r)
    ur_norm = u / np.sqrt(norm)
    return ur_norm

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.10', 3)
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
assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 2, Z, hu), target)
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
assert np.allclose(compute_Schrod_Hartree(-0.4, r_grid, 2, Z, hu), target)
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
assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 3, Z, hu), target)
