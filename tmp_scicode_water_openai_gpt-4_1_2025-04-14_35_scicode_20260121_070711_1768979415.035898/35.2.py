import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    hbar = 1.054571817e-34
    c = 299792458
    pi = 3.141592653589793
    E1 = (pi**2 * hbar**2) / (2 * mr * L**2)
    wavelength = (hbar * c * 2 * pi) / E1
    return wavelength
def calculate_force(mass, acceleration):
    return mass * acceleration

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.2', 3)
target = targets[0]

C = generate_quadratic_combinations(7, 11, 13, 5)
assert np.allclose(sorted(C), target)
target = targets[1]

C = generate_quadratic_combinations(7, 11, 13, 10)
assert np.allclose(sorted(C), target)
target = targets[2]

C = generate_quadratic_combinations(71, 19, 17, 5)
assert np.allclose(sorted(C), target)
