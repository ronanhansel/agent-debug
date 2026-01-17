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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.3', 3)
target = targets[0]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,10,20), 1), target)
target = targets[1]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,10), 2), target)
target = targets[2]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,20), 3), target)
