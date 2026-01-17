import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(E0, x, y, z, wavelength):
    k = 2 * np.pi / wavelength
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx, Ny = len(x), len(y)
    fx = np.fft.fftfreq(Nx, dx)
    fy = np.fft.fftfreq(Ny, dy)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    H = np.exp(1j * k * z * np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2))
    E0_fft = np.fft.fft2(E0)
    E_fft = E0_fft * H
    return np.fft.ifft2(E_fft)
def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    q_initial = 1.0 / (1.0 / R0 - 1j * wavelength / (np.pi * w0**2))
    q0 = Mp2 * q_initial
    def M_fs(d):
        return np.array([[1.0, d], [0.0, 1.0]], dtype=float)
    M_s = M_fs(s)
    M_L1 = M_fs(L1)
    M_pre = np.dot(np.dot(M_L1, Mf1), M_s)
    wz = np.zeros_like(z, dtype=float)
    for idx, zi in enumerate(z):
        M_z = M_fs(zi)
        M_tot = np.dot(M_z, M_pre)
        A, B = M_tot[0, 0], M_tot[0, 1]
        C, D = M_tot[1, 0], M_tot[1, 1]
        q_z = (A * q0 + B) / (C * q0 + D)
        wz[idx] = np.sqrt(-wavelength / (np.pi * np.imag(1.0 / q_z)))
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
