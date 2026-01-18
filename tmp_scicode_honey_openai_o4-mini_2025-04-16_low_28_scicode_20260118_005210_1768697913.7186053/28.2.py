import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    import numpy as np
    dx = L / N
    x = np.linspace(-L/2, L/2, N+1)
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X**2 + Y**2) / w0**2)
    Gau = np.abs(E0)
    fx = np.fft.fftfreq(N+1, d=dx)
    kx = 2 * np.pi * fx
    KX, KY = np.meshgrid(kx, kx)
    k = 2 * np.pi / Ld
    H = np.exp(-1j * (KX**2 + KY**2) * z / (2 * k))
    E0_hat = np.fft.fft2(E0)
    E_pro = np.fft.ifft2(E0_hat * H)
    Gau_pro = np.abs(E_pro)
    return Gau, Gau_pro
def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    z = np.array(z, dtype=float)
    q_inv = 1.0 / R0 - 1j * wavelength / (np.pi * w0**2)
    q0 = Mp2 * (1.0 / q_inv)
    def M_free(d):
        return np.array([[1.0, d],
                         [0.0, 1.0]], dtype=complex)
    M_total = M_free(L1) @ Mf1 @ M_free(s)
    wz = np.zeros_like(z, dtype=float)
    for idx, zi in enumerate(z):
        Mzi = M_free(zi) @ M_total
        A, B = Mzi[0, 0], Mzi[0, 1]
        C, D = Mzi[1, 0], Mzi[1, 1]
        qz = (A * q0 + B) / (C * q0 + D)
        inv_qz = 1.0 / qz
        wz[idx] = np.sqrt(-wavelength / (np.pi * np.imag(inv_qz)))
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
