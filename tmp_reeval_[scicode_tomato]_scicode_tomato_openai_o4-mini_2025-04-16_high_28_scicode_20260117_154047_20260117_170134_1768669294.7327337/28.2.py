import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    """
    Propagate a Gaussian beam and calculate its amplitude distribution
    before and after propagation using the angular spectrum method.

    Parameters
    ----------
    N : int
        Number of sampling points per side (grid will be (N+1)x(N+1)).
    Ld : float
        Wavelength of the beam.
    w0 : float
        Beam waist radius at z = 0.
    z : float
        Propagation distance.
    L : float
        Physical side length of the square sampling window.

    Returns
    -------
    Gau : ndarray, shape (N+1, N+1)
        Absolute value of the field amplitude at z = 0.
    Gau_pro : ndarray, shape (N+1, N+1)
        Absolute value of the field amplitude after propagation distance z.
    """
    dx = L / N
    x = np.linspace(-L/2, L/2, N+1)
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X**2 + Y**2) / w0**2)
    E0_F = np.fft.fftshift(np.fft.fft2(E0))
    fx = np.fft.fftshift(np.fft.fftfreq(N+1, d=dx))
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * Ld * z * (FX**2 + FY**2))
    E_prop_F = E0_F * H
    E_prop = np.fft.ifft2(np.fft.ifftshift(E_prop_F))
    Gau = np.abs(E0)
    Gau_pro = np.abs(E_prop)
    return Gau, Gau_pro
def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    """
    Simulate Gaussian beam propagation through one lens (Mf1) plus free‐space segments,
    computing the beam waist radius at each distance in z.

    Parameters
    ----------
    wavelength : float
        Wavelength of the light [m].
    w0 : float
        Initial beam waist radius [m].
    R0 : float
        Radius of curvature of the wavefront at input [m] (use np.inf for a plane wave).
    Mf1 : ndarray, shape (2,2)
        ABCD matrix of the first optical element.
    z : ndarray
        1D array of distances from lens where waist is calculated [m].
    Mp2 : float
        Scaling factor for the initial complex beam parameter.
    L1 : float
        Distance from the first element to the measurement region [m].
    s : float
        Distance from source to the first optical element [m].

    Returns
    -------
    wz : ndarray
        Beam waist radius at each z [m].
    """
    inv_R0 = 0.0 if np.isinf(R0) else 1.0 / R0
    q0_inv = inv_R0 - 1j * wavelength / (np.pi * w0**2)
    q0 = Mp2 * (1.0 / q0_inv)
    M_s = np.array([[1.0, s], [0.0, 1.0]])
    M_L1 = np.array([[1.0, L1], [0.0, 1.0]])
    M_after = M_L1 @ Mf1 @ M_s
    wz = np.zeros_like(z, dtype=float)
    for idx, zi in enumerate(z):
        M_z = np.array([[1.0, zi], [0.0, 1.0]])
        A, B, C, D = (M_z @ M_after).ravel()[:4]
        q_out = (A * q0 + B) / (C * q0 + D)
        wz[idx] = np.sqrt(-wavelength / (np.pi * np.imag(1.0 / q_out)))
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
