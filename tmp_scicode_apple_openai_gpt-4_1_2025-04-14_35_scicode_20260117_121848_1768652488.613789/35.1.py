import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    '''
    Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    m_e = 9.109e-31
    c = 3e8
    h = 6.626e-34
    hbar = h / (2 * 3.141592653589793)
    L_m = L * 1e-9
    m_eff = mr * m_e
    E1 = (3.141592653589793**2) * (hbar**2) / (2 * m_eff * L_m**2)
    lmbd_m = h * c / E1
    lmbd = lmbd_m * 1e9
    return lmbd

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.1', 3)
target = targets[0]

assert np.allclose(ground_state_wavelength(5,0.6), target)
target = targets[1]

assert np.allclose(ground_state_wavelength(10,0.6), target)
target = targets[2]

assert np.allclose(ground_state_wavelength(10,0.06), target)
