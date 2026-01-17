# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    f_r = []
    eps = 1e-15
    for r in r_grid:
        r_safe = r if r != 0.0 else eps
        f_val = (l * (l + 1)) / (r_safe ** 2) - 2.0 / r_safe - 2.0 * energy
        f_r.append(f_val)
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    '''
    Solve u''(r) = f(r)·u(r) on an equally-spaced radial grid by the Numerov method.
    Parameters
    ----------
    f_in : (N,) array_like
        Values of f(r) on a uniform grid r = n*step.
    u_at_0 : float
        Boundary condition u(r=0).
    up_at_0 : float
        Boundary condition u'(r=0).
    step : float
        Grid spacing h.
    Returns
    -------
    u : list, length N
        Numerical solution u(r) on the same grid.
    '''
    f_in = [float(v) for v in f_in]
    N = len(f_in)
    h = float(step)
    u = [0.0] * N
    u[0] = u_at_0
    if N == 1:
        return u
    h2 = h * h
    u[1] = u_at_0 + h * up_at_0 + 0.5 * h2 * f_in[0] * u_at_0
    if N == 2:
        return u
    h2_over_12 = h2 / 12.0
    for i in range(1, N - 1):
        k_im1 = f_in[i - 1]
        k_i = f_in[i]
        k_ip1 = f_in[i + 1]
        numerator = (2.0 * u[i] * (1.0 - 5.0 * h2_over_12 * k_i)
                     - u[i - 1] * (1.0 + h2_over_12 * k_im1))
        denominator = 1.0 + h2_over_12 * k_ip1
        u[i + 1] = numerator / denominator
    return u

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.2', 3)
target = targets[0]

assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[1]

assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[2]

assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
