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
def generate_quadratic_combinations(x, y, z, N):
    margin = 6
    found_enough = False
    M = int(N ** (1/3)) + margin

    while not found_enough:
        Q_set = set()
        for i in range(1, M+1):
            for j in range(1, M+1):
                for k in range(1, M+1):
                    q = i**2 * x + j**2 * y + k**2 * z
                    Q_set.add(q)
        Q_sorted = sorted(Q_set)
        if len(Q_sorted) >= N:
            found_enough = True
        else:
            M += margin

    C = Q_sorted[:N]
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
