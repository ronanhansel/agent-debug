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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.2', 3)
target = targets[0]

assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[1]

assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[2]

assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
