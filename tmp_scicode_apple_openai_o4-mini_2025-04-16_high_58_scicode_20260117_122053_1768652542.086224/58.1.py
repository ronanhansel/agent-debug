import numpy as np
import scipy as sp
import scipy.integrate as si

def required_function():
    pass

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('58.1', 3)
target = targets[0]

rho = 0.1
eos_Gamma = 2.0
eos_kappa = 100.
assert np.allclose(eos_press_from_rho(rho, eos_Gamma, eos_kappa), target)
target = targets[1]

rho = 0.2
eos_Gamma = 3./5.
eos_kappa = 80
assert np.allclose(eos_press_from_rho(rho, eos_Gamma, eos_kappa), target)
target = targets[2]

rho = 1.1
eos_Gamma = 1.8
eos_kappa = 20
assert np.allclose(eos_press_from_rho(rho, eos_Gamma, eos_kappa), target)
