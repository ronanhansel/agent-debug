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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
