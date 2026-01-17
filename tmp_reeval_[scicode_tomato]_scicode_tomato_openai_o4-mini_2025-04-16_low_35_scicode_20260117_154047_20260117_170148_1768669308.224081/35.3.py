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
def absorption(mr, a, b, c, N):
    if N <= 0:
        return []
    m_e = 9.109e-31
    h = 6.626e-34
    c0 = 3e8
    m_eff = mr * m_e
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9
    E1a = h**2 / (8 * m_eff * a_m * a_m)
    E1b = h**2 / (8 * m_eff * b_m * b_m)
    E1c = h**2 / (8 * m_eff * c_m * c_m)
    E0 = E1a + E1b + E1c
    vals = []
    for i in range(1, N + 2):
        for j in range(1, N + 2):
            for k in range(1, N + 2):
                vals.append(i * i * E1a + j * j * E1b + k * k * E1c)
    vals.sort()
    combos = vals[:N + 1]
    delta_E = [v - E0 for v in combos]
    excited = delta_E[1:N + 1]
    wavelengths = [(h * c0 / e) * 1e9 for e in excited]
    wavelengths.sort(reverse=True)
    return wavelengths

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.3', 4)
target = targets[0]

A = absorption(0.6,3,4,10**6,5)
assert (all(i>10**10 for i in A)) == target
target = targets[1]

A = absorption(0.3,7,3,5,10)
assert np.allclose(sorted(A)[::-1], target)
target = targets[2]

A = absorption(0.6,3,4,5,5)
assert np.allclose(sorted(A)[::-1], target)
target = targets[3]

A = absorption(0.6,37,23,18,10)
assert np.allclose(sorted(A)[::-1], target)
