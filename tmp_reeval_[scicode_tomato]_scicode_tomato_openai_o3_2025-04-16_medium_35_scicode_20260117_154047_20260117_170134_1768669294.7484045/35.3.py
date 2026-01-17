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
def absorption(mr, a, b, c, N):
    '''Return the N largest photon wavelengths (nm) corresponding to the
    smallest N non-zero transition energies of an electron (effective mass
    mr·m_e) confined in a 3-D cuboid infinite potential well whose side
    lengths are a, b, and c (all in nanometres).

    Parameters
    ----------
    mr : float
        Relative effective electron mass (must be > 0).
    a, b, c : float
        Side lengths of the quantum dot in nanometres (each > 0).
    N : int
        Number of transition wavelengths required.

    Returns
    -------
    numpy.ndarray
        1-D array (length N) containing wavelengths in nanometres sorted in
        descending order (largest λ first = smallest ΔE).
    '''
    # --------------------- input validation ---------------------------------
    if N <= 0:
        return np.array([], dtype=float)
    if mr <= 0 or a <= 0 or b <= 0 or c <= 0:
        raise ValueError("mr, a, b, and c must all be positive.")

    # --------------------- physical constants -------------------------------
    m_e = 9.109e-31         # kg
    c_light = 3.0e8         # m/s
    h = 6.626e-34           # J·s

    # --------------------- convert lengths to metres ------------------------
    a_m, b_m, c_m = np.array([a, b, c], dtype=float) * 1e-9

    # --------------------- effective mass -----------------------------------
    m_eff = mr * m_e

    # --------------------- base energies (n=1 in each axis) -----------------
    factor = h**2 / (8.0 * m_eff)
    E_a = factor / (a_m**2)
    E_b = factor / (b_m**2)
    E_c = factor / (c_m**2)

    # --------------------- enumerate transition energies --------------------
    energies = set()
    L = 2  # start at 2; (1,1,1) gives ΔE = 0
    while len(energies) < N:
        for i, j, k in itertools.product(range(1, L + 1), repeat=3):
            dE = (i * i - 1) * E_a + (j * j - 1) * E_b + (k * k - 1) * E_c
            if dE > 0:
                energies.add(dE)
        if len(energies) < N:
            L += 1

    # --------------------- select N smallest energies -----------------------
    smallest_E = np.array(sorted(energies)[:N], dtype=float)

    # --------------------- convert to wavelengths (nm) ----------------------
    wavelengths_nm = (h * c_light) / smallest_E / 1e-9  # metres → nanometres

    # --------------------- return descending order --------------------------
    return np.sort(wavelengths_nm)[::-1]

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
