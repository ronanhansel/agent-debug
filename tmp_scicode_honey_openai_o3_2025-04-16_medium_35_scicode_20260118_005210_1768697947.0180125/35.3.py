import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    def is_iterable(obj):
        try:
            iter(obj)
            return not isinstance(obj, (str, bytes))
        except TypeError:
            return False

    def compute_lambda(l_val, m_rel):
        m_e = 9.109e-31
        h = 6.626e-34
        c = 3.0e8
        L_m = l_val * 1e-9
        m_eff = m_rel * m_e
        E1 = h ** 2 / (8.0 * m_eff * L_m ** 2)
        return (h * c) / E1 * 1e9

    if not is_iterable(L) and not is_iterable(mr):
        return compute_lambda(float(L), float(mr))

    if is_iterable(L) and not is_iterable(mr):
        mr_iter = [mr] * len(L)
    elif not is_iterable(L) and is_iterable(mr):
        L = [L] * len(mr)
        mr_iter = mr
    else:
        mr_iter = mr

    return [compute_lambda(float(l), float(m_rel)) for l, m_rel in zip(L, mr_iter)]
def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array of length N that contains the
    smallest N distinct values of i^2·x + j^2·y + k^2·z, where i, j, k are
    positive integers (≥ 1).  The values are returned in ascending order.

    Parameters
    ----------
    x : float
        First positive number.
    y : float
        Second positive number.
    z : float
        Third positive number.
    N : int
        How many of the smallest quadratic–combination values to return.

    Returns
    -------
    list
        1-D list (length N) of the smallest distinct quadratic combinations.
    '''
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError("x, y, z must all be positive.")
    x, y, z = float(x), float(y), float(z)
    N = int(N)
    limit = 1
    while limit ** 3 < N:
        limit += 1
    unique_vals = set()
    while True:
        for i in range(1, limit + 1):
            ii = i * i
            for j in range(1, limit + 1):
                jj = j * j
                for k in range(1, limit + 1):
                    val = ii * x + jj * y + (k * k) * z
                    unique_vals.add(round(val, 12))
        if len(unique_vals) >= N:
            break
        limit *= 2
    smallest_vals = sorted(unique_vals)[:N]
    return smallest_vals
def absorption(mr, a, b, c, N):
    # validate inputs
    if mr <= 0 or a <= 0 or b <= 0 or c <= 0:
        raise ValueError('mr, a, b, and c must all be positive.')
    if not (isinstance(N, int) and N > 0):
        raise ValueError('N must be a positive integer.')
    # physical constants
    m_e = 9.109e-31
    h = 6.626e-34
    c_vel = 3.0e8
    # convert geometry to metres
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9
    # energy coefficients
    m_eff = mr * m_e
    pref = h ** 2 / (8.0 * m_eff)
    x = pref / (a_m ** 2)
    y = pref / (b_m ** 2)
    z = pref / (c_m ** 2)
    E0 = x + y + z
    # gather smallest positive transition energies
    transitions = set()
    limit = 1
    while len(transitions) < N:
        for i in range(1, limit + 1):
            for j in range(1, limit + 1):
                for k in range(1, limit + 1):
                    Etot = (i * i) * x + (j * j) * y + (k * k) * z
                    dE = Etot - E0
                    if dE > 0:
                        transitions.add(round(dE, 15))
        limit += 1
    dE_smallest = sorted(transitions)[:N]
    wavelengths_nm = [(h * c_vel / dE) * 1e9 for dE in dE_smallest]
    return sorted(wavelengths_nm, reverse=True)

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
