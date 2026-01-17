# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    safe_r = r_grid.copy()
    safe_r[safe_r == 0] = 1e-12
    f_r = -2.0 / safe_r + l * (l + 1) / (safe_r ** 2) - 2.0 * energy
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    N = len(f_in)
    h = step
    u = [0.0]*N
    u[0] = u_at_0
    if N > 1:
        u[1] = u_at_0 + h * up_at_0 + 0.5 * h * h * f_in[0] * u_at_0
    for n in range(1, N - 1):
        h2 = h * h
        denom = 1 + (h2 / 12.0) * f_in[n + 1]
        num = (2 * u[n] * (1 - (5 * h2 / 12.0) * f_in[n])
               - u[n - 1] * (1 + (h2 / 12.0) * f_in[n - 1]))
        u[n + 1] = num / denom
    return u

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.2', 3)
target = targets[0]

assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[1]

assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[2]

assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
