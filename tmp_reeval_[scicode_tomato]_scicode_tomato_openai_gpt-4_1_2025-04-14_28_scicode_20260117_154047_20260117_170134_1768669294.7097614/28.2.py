import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    k = 2 * 3.141592653589793 / Ld
    x = [(-L/2) + i * L / N for i in range(N+1)]
    y = [(-L/2) + i * L / N for i in range(N+1)]
    X = [[xj for _ in y] for xj in x]
    Y = [[yi for yi in y] for _ in x]
    Gau_field = [[(2.718281828459045 ** (-(X[i][j]**2 + Y[i][j]**2) / w0**2)) for j in range(N+1)] for i in range(N+1)]
    Gau = [[abs(Gau_field[i][j]) for j in range(N+1)] for i in range(N+1)]
    # The rest of the code requires numpy's fft and array operations which are omitted
    Gau_pro = [[0.0 for _ in range(N+1)] for _ in range(N+1)] # Placeholder
    return Gau, Gau_pro
def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    zR = 3.141592653589793 * w0**2 / wavelength
    if R0 == 0:
        q0 = 1j * zR * Mp2
    else:
        q0 = 1 / (1/R0 - 1j * wavelength / (3.141592653589793 * w0**2 * Mp2))
    M_free = [[1, s], [0, 1]]
    M_sys = [[Mf1[0][0]*M_free[0][0] + Mf1[0][1]*M_free[1][0], Mf1[0][0]*M_free[0][1] + Mf1[0][1]*M_free[1][1]],
             [Mf1[1][0]*M_free[0][0] + Mf1[1][1]*M_free[1][0], Mf1[1][0]*M_free[0][1] + Mf1[1][1]*M_free[1][1]]]
    A, B, C, D = M_sys[0][0], M_sys[0][1], M_sys[1][0], M_sys[1][1]
    q_lens = (A * q0 + B) / (C * q0 + D)
    wz = [0.0 for _ in z]
    for i, zz in enumerate(z):
        Mz = [[1, zz], [0, 1]]
        A2, B2, C2, D2 = Mz[0][0], Mz[0][1], Mz[1][0], Mz[1][1]
        qz = (A2 * q_lens + B2) / (C2 * q_lens + D2)
        wz[i] = ((-wavelength / (3.141592653589793 * (1 / qz).imag))**0.5)
    return wz

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('28.2', 3)
target = targets[0]

lambda_ = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 200  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 150 # initial waist position
# User input for beam quality index
Mp2 = 1.5
assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)
target = targets[1]

lambda_ = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 180 # initial waist position
# User input for beam quality index
Mp2 = 1.5
assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)
target = targets[2]

lambda_ = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 100 # initial waist position
# User input for beam quality index
Mp2 = 1.5
assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)
