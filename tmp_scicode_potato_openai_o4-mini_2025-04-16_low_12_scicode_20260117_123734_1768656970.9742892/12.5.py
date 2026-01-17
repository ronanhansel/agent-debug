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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.5', 3)
target = targets[0]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),2, -1.2/np.arange(1,20,0.2)**2), target)
target = targets[1]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),3,-1.2/np.arange(1,20,0.2)**2), target)
target = targets[2]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),0,-1.2/np.arange(1,20,0.2)**2), target)
