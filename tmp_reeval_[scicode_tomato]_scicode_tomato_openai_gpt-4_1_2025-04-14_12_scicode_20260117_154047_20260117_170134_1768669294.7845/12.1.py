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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
