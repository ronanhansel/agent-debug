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
def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array of length N containing
    the smallest N values of i^2*x + j^2*y + k^2*z for i,j,k ≥ 1, sorted ascending.'''
    if N <= 0:
        return np.array([], dtype=float)
    values = [
        i*i*x + j*j*y + k*k*z
        for i, j, k in itertools.product(range(1, N+1), repeat=3)
    ]
    values.sort()
    return np.array(values[:N], dtype=float)

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
