import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    """Given the width of an infinite square well (nm) and relative effective mass mr,
    return the wavelength (nm) corresponding to the ground-state energy."""
    m_e = 9.109e-31
    h = 6.626e-34
    c = 3e8
    L_m = L * 1e-9
    m_eff = mr * m_e
    E = h**2 / (8 * m_eff * L_m**2)
    lambda_m = h * c / E
    lmbd = lambda_m * 1e9
    return lmbd

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.1', 3)
target = targets[0]

assert np.allclose(ground_state_wavelength(5,0.6), target)
target = targets[1]

assert np.allclose(ground_state_wavelength(10,0.6), target)
target = targets[2]

assert np.allclose(ground_state_wavelength(10,0.06), target)
