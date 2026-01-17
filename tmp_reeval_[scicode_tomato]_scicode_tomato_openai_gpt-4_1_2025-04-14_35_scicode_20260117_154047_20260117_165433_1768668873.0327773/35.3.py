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
def absorption(mr, a, b, c, N):
    me = 9.109e-31
    h = 6.626e-34
    c_light = 3e8
    mstar = mr * me
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9
    E_a = h**2 / (8 * mstar * a_m**2)
    E_b = h**2 / (8 * mstar * b_m**2)
    E_c = h**2 / (8 * mstar * c_m**2)
    energies = set()
    max_n = max(4, int((N*2)**0.5) + 3)
    while True:
        for i, j, k in itertools.product(range(1, max_n+1), repeat=3):
            E = i**2 * E_a + j**2 * E_b + k**2 * E_c
            energies.add(E)
        energies_sorted = sorted(energies)
        ground = energies_sorted[0]
        transitions = sorted(set(e - ground for e in energies_sorted[1:] if e - ground > 0))
        if len(transitions) >= N:
            break
        else:
            max_n += 2
    required_transitions = transitions[:N]
    wavelengths = [(h * c_light) / E * 1e9 for E in required_transitions]
    wavelengths.sort(reverse=True)
    return np.array(wavelengths)

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
