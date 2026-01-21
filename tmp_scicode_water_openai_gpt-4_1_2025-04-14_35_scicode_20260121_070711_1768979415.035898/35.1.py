import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    hbar = 1.054571817e-34
    c = 299792458
    pi = 3.141592653589793
    E1 = (pi**2 * hbar**2) / (2 * mr * L**2)
    wavelength = (hbar * c * 2 * pi) / E1
    return wavelength

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.1', 3)
target = targets[0]

assert np.allclose(ground_state_wavelength(5,0.6), target)
target = targets[1]

assert np.allclose(ground_state_wavelength(10,0.6), target)
target = targets[2]

assert np.allclose(ground_state_wavelength(10,0.06), target)
