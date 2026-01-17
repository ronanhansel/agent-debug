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
def absorption(mr, a, b, c, N):
    m_e = 9.109e-31
    c_light = 3e8
    h = 6.626e-34
    hbar = h / (2 * 3.141592653589793)
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9
    m_eff = mr * m_e
    const = (3.141592653589793 ** 2) * (hbar ** 2) / (2 * m_eff)
    margin = 6
    M = int((N**(1/3)) + margin)
    indices = range(1, M+1)
    states = []
    for i in indices:
        for j in indices:
            for k in indices:
                states.append((i, j, k))
    Xs = []
    for i, j, k in states:
        Xs.append((i**2 / a_m**2) + (j**2 / b_m**2) + (k**2 / c_m**2))
    X111 = (1 / a_m**2) + (1 / b_m**2) + (1 / c_m**2)
    delta_Xs = []
    for x in Xs:
        delta_Xs.append(x - X111)
    mask = []
    for dx in delta_Xs:
        mask.append(dx > 0)
    selected_Xs = []
    for i, dx in enumerate(delta_Xs):
        if mask[i]:
            selected_Xs.append(dx)
    E_trans = []
    for dx in selected_Xs:
        E_trans.append(const * dx)
    mask2 = []
    for e in E_trans:
        mask2.append(e > 0)
    filtered_E_trans = []
    for i, e in enumerate(E_trans):
        if mask2[i]:
            filtered_E_trans.append(e)
    wavelengths_nm = []
    for e in filtered_E_trans:
        wavelengths_nm.append((h * c_light / e) * 1e9)
    wavelengths_unique = sorted(set(wavelengths_nm), reverse=True)
    if len(wavelengths_unique) < N:
        wavelengths_out = [0.]*N
        for i, wl in enumerate(wavelengths_unique):
            wavelengths_out[i] = wl
    else:
        wavelengths_out = wavelengths_unique[:N]
    return wavelengths_out

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
