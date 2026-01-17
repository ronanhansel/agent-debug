# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    m_e=9.1093837015e-31
    hbar=1.054571817e-34
    e_charge=1.602176634e-19
    eps0=8.8541878128e-12
    pi=3.141592653589793
    centrifugal=l*(l+1)/(r_grid**2)
    V_coulomb=-e_charge**2/(4.0*pi*eps0*r_grid)
    prefactor=2.0*m_e/(hbar**2)
    f_r=centrifugal+prefactor*(V_coulomb-energy)
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    """
    Given precomputed function f(r), solve the differential equation u''(r) = f(r)*u(r)
    using the Numerov method.
    Inputs:
    - f_in: 1D array of float representing f(r) at uniform grid points.
    - u_at_0: the value u(0), a float.
    - up_at_0: the derivative u'(0), a float.
    - step: the grid spacing Δr, a float.
    Output:
    - u: 1D array of float with the solution u(r) at each grid point.
    """
    N = len(f_in)
    u = [0.0 for _ in range(N)]
    u[0] = u_at_0
    if N > 1:
        u[1] = u_at_0 + step * up_at_0 + 0.5 * (step**2) * f_in[0] * u_at_0
    k = [1.0 + (step**2 / 12.0) * fi for fi in f_in]
    for n in range(1, N - 1):
        u[n+1] = (2.0 * k[n] * u[n] - k[n-1] * u[n-1]) / k[n+1]
    return u
def compute_Schrod(energy, r_grid, l):
    """
    Solve the radial Schrödinger equation u'' = f(r) u with Numerov's method
    and normalize the solution by Simpson's rule (integrating from largest r).
    
    Inputs:
      - energy: float, the energy eigenvalue E
      - r_grid: 1D numpy array of floats, the radial grid points
      - l: int, angular momentum quantum number
    Output:
      - ur_norm: 1D numpy array of floats, the normalized radial wavefunction u(r)
    """
    step = r_grid[0] - r_grid[1]
    f_r = f_Schrod(energy, l, r_grid)
    u_vals = Numerov(f_r, u_at_0=0.0, up_at_0=-1e-7, step=step)
    u = np.array(u_vals, dtype=float)
    if hasattr(integrate, 'simps'):
        simpson_func = integrate.simps
    else:
        simpson_func = integrate.simpson
    r_rev = r_grid[::-1]
    u_rev = u[::-1]
    integral = simpson_func(u_rev**2, r_rev)
    norm_const = 1.0 / np.sqrt(integral)
    ur_norm = u * norm_const
    return ur_norm

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.3', 3)
target = targets[0]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,10,20), 1), target)
target = targets[1]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,10), 2), target)
target = targets[2]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,20), 3), target)
