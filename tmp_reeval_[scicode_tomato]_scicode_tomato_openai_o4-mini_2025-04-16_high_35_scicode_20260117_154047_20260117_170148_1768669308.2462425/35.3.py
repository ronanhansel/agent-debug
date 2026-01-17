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
def absorption(mr, a, b, c, N):
    '''With the feature sizes in three dimensions a, b, and c, the relative mass mr and the array length N, return a numpy array of the size N that contains the corresponding photon wavelengths (nm) of the excited‐state transitions.
    Input:
    mr (float): relative effective electron mass.
    a (float): Feature size in the first dimension (nm).
    b (float): Feature size in the second dimension (nm).
    c (float): Feature size in the third dimension (nm).
    N (int): The number of transition wavelengths to return.
    Output:
    A (numpy.ndarray of shape (N,)): The N largest wavelengths (nm), in descending order.
    '''
    if mr <= 0 or a <= 0 or b <= 0 or c <= 0:
        raise ValueError("mr, a, b, and c must be positive.")
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer.")
    m_e = 9.109e-31
    h = 6.626e-34
    c0 = 3e8
    a_m, b_m, c_m = a*1e-9, b*1e-9, c*1e-9
    m_eff = mr * m_e
    E1_x = h*h / (8 * m_eff * a_m*a_m)
    E1_y = h*h / (8 * m_eff * b_m*b_m)
    E1_z = h*h / (8 * m_eff * c_m*c_m)
    E1_total = E1_x + E1_y + E1_z
    M = int((N+1)**(1/3))
    while M**3 < N + 1:
        M += 1
    wavelengths = []
    for i, j, k in itertools.product(range(1, M+1), repeat=3):
        if (i, j, k) == (1, 1, 1):
            continue
        E_ijk = E1_x*(i*i) + E1_y*(j*j) + E1_z*(k*k)
        delta_E = E_ijk - E1_total
        if delta_E > 0:
            lambda_m = h * c0 / delta_E
            wavelengths.append(lambda_m * 1e9)
    wavelengths.sort(reverse=True)
    return np.array(wavelengths[:N])

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
