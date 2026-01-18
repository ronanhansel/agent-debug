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
def absorption(mr, a, b, c, N):
    """
    With the feature sizes in three dimensions a, b, and c (nm), the relative mass mr
    and the array length N, return a numpy array of the size N that contains the
    photon wavelengths (nm) of the smallest N non-zero transition energies.
    The output is sorted in descending order (largest λ first).
    """
    m_e = 9.109e-31
    h = 6.626e-34
    c_light = 3.0e8
    m_eff = mr * m_e
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9
    E_x = h**2 / (8 * m_eff * a_m**2)
    E_y = h**2 / (8 * m_eff * b_m**2)
    E_z = h**2 / (8 * m_eff * c_m**2)
    E0 = E_x + E_y + E_z
    combined = generate_quadratic_combinations(E_x, E_y, E_z, N + 1)
    E_trans = combined[1:] - E0
    lambdas_nm = (h * c_light / E_trans) * 1e9
    return np.sort(lambdas_nm)[::-1]

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
