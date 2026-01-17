import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    x = np.linspace(-L / 2, L / 2, N + 1)
    y = np.linspace(-L / 2, L / 2, N + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    E0 = np.exp(- (X**2 + Y**2) / (w0**2))
    Gau = np.abs(E0)
    dx = L / N
    fx = np.fft.fftfreq(N + 1, d=dx)
    fy = np.fft.fftfreq(N + 1, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    k = 2 * np.pi / Ld
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz_squared = k**2 - kx**2 - ky**2
    kz = np.sqrt(kz_squared + 0j)
    H = np.exp(1j * kz * z)
    E0_ft = np.fft.fft2(E0)
    E1_ft = E0_ft * H
    E1 = np.fft.ifft2(E1_ft)
    Gau_pro = np.abs(E1)
    return Gau, Gau_pro
def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    if float('inf') == R0:
        q0 = 1j * 3.141592653589793 * w0**2 / (wavelength * Mp2)
    else:
        q0 = 1 / (1/R0 - 1j * wavelength / (3.141592653589793 * w0**2) * Mp2)
    Ms = [[1, s], [0, 1]]
    A, B, C, D = Ms[0][0], Ms[0][1], Ms[1][0], Ms[1][1]
    q1 = (A * q0 + B) / (C * q0 + D)
    A, B, C, D = Mf1[0][0], Mf1[0][1], Mf1[1][0], Mf1[1][1]
    q2 = (A * q1 + B) / (C * q1 + D)
    wz = [0.0 for _ in range(len(z))]
    for i, zi in enumerate(z):
        Af, Bf, Cf, Df = 1, zi, 0, 1
        qz = (Af * q2 + Bf) / (Cf * q2 + Df)
        inv_qz = 1 / qz
        wz[i] = ((-wavelength / (3.141592653589793 * (inv_qz.imag)))**0.5)
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
