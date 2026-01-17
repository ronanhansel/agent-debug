import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    if L <= 0 or mr <= 0:
        raise ValueError("Well width L and relative mass mr must be positive.")
    m_e = 9.109e-31
    h = 6.626e-34
    c = 3e8
    L_m = L * 1e-9
    m = mr * m_e
    E1 = h**2 / (8 * m * L_m**2)
    lambda_m = h * c / E1
    return lambda_m * 1e9
def generate_quadratic_combinations(x, y, z, N):
    """
    With three positive numbers x, y, z, return a list of length N containing 
    the smallest N values of the form i^2*x + j^2*y + k^2*z, where i, j, k are 
    positive integers starting from 1.
    """
    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError("x, y, and z must be positive.")
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer.")
    M = int(N ** (1/3))
    while M**3 < N:
        M += 1
    i2x = [i * i * x for i in range(1, M + 1)]
    j2y = [j * j * y for j in range(1, M + 1)]
    k2z = [k * k * z for k in range(1, M + 1)]
    E = []
    for i_val in i2x:
        for j_val in j2y:
            for k_val in k2z:
                E.append(i_val + j_val + k_val)
    E.sort()
    return E[:N]

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
