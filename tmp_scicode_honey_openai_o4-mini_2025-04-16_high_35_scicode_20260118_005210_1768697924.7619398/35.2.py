import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    """
    Given the width of an infinite square well, provide the corresponding wavelength
    of the ground state eigen-state energy.

    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): Relative effective electron mass.

    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    """
    m_e = 9.109e-31
    h = 6.626e-34
    c = 3.0e8
    L_m = L * 1e-9
    m_eff = mr * m_e
    E1 = h**2 / (8 * m_eff * L_m**2)
    lambda_m = h * c / E1
    lmbd = lambda_m * 1e9
    return lmbd
def generate_quadratic_combinations(x, y, z, N):
    """
    With three numbers given, return a numpy array of length N containing the
    smallest N values of i^2*x + j^2*y + k^2*z with integer i, j, k >= 1.
    Duplicates (degeneracies) are included and the array is in ascending order.
    """
    visited = {(1, 1, 1)}
    candidates = [(1, 1, 1)]
    results = []

    def Q(triple):
        i, j, k = triple
        return i*i*x + j*j*y + k*k*z

    while len(results) < N:
        min_idx = 0
        min_val = Q(candidates[0])
        for idx in range(1, len(candidates)):
            val = Q(candidates[idx])
            if val < min_val:
                min_val = val
                min_idx = idx
        i, j, k = candidates.pop(min_idx)
        results.append(min_val)
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nbr = (i + di, j + dj, k + dk)
            if nbr not in visited:
                visited.add(nbr)
                candidates.append(nbr)
    return np.array(results)

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
