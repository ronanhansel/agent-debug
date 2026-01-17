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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.2', 3)
target = targets[0]

assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[1]

assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[2]

assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
