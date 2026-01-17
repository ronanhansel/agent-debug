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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
