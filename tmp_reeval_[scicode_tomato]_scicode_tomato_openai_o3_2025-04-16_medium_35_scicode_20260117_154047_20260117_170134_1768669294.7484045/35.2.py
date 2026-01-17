import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    m_e = 9.109e-31
    c = 3.0e8
    h = 6.626e-34
    L_m = L * 1e-9
    m_eff = mr * m_e
    E1 = h**2 / (8.0 * m_eff * L_m**2)
    lambda_m = (h * c) / E1
    lmbd = lambda_m / 1e-9
    return lmbd
def generate_quadratic_combinations(x, y, z, N):
    if N <= 0:
        return []
    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError('Inputs x, y, and z must all be positive.')
    combos = set()
    L = 1
    while len(combos) < N:
        for i in range(1, L + 1):
            i_sq = i * i
            for j in range(1, L + 1):
                j_sq = j * j
                for k in range(1, L + 1):
                    k_sq = k * k
                    combos.add(i_sq * x + j_sq * y + k_sq * z)
        if len(combos) < N:
            L += 1
    return sorted(combos)[:N]

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
