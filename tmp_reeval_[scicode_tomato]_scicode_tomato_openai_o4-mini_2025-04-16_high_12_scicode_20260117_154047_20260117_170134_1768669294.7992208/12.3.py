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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.3', 3)
target = targets[0]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,10,20), 1), target)
target = targets[1]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,10), 2), target)
target = targets[2]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,20), 3), target)
