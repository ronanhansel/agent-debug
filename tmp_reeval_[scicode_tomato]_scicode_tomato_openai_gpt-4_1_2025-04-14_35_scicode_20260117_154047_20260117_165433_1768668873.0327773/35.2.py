import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    me = 9.109e-31
    c = 3e8
    h = 6.626e-34
    mstar = mr * me
    L_m = L * 1e-9
    hbar = h / (2 * 3.141592653589793)
    E1 = (3.141592653589793 ** 2) * (hbar ** 2) / (2 * mstar * (L_m ** 2))
    lmbd = (h * c) / E1 * 1e9
    return lmbd
def generate_quadratic_combinations(x, y, z, N):
    combinations_set = set()
    M = max(int((N * 2)**0.5 + 0.999999), 3)
    found_enough = False
    while not found_enough:
        for i in range(1, int(M) + 1):
            for j in range(1, int(M) + 1):
                for k in range(1, int(M) + 1):
                    val = i ** 2 * x + j ** 2 * y + k ** 2 * z
                    combinations_set.add(val)
        if len(combinations_set) >= N:
            found_enough = True
        else:
            M += 2
    sorted_combos = sorted(list(combinations_set))
    C = sorted_combos[:N]
    return C

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
