import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    '''
    Propagate a Gaussian beam and calculate its intensity distribution before and after propagation.

    Parameters
    ----------
    N  : int
        Number of sampling points per spatial dimension (square grid).  Output
        arrays have shape (N+1, N+1).
    Ld : float
        Wavelength λ of the Gaussian beam.
    w0 : float
        Beam-waist radius at z = 0 (1/e field radius).
    z  : float
        Propagation distance.
    L  : float
        Physical side length of the square sampling window.

    Returns
    -------
    Gau : ndarray, shape (N+1, N+1)
        Magnitude of the beam’s amplitude distribution at z = 0.
    Gau_pro : ndarray, shape (N+1, N+1)
        Magnitude of the beam’s amplitude distribution after propagating
        the distance z.
    '''
    n_pts = N + 1
    dx = L / N
    x = np.linspace(-L / 2.0, L / 2.0, n_pts)
    X, Y = np.meshgrid(x, x, indexing='xy')
    E0 = np.exp(-(X**2 + Y**2) / (w0 ** 2))
    Gau = np.abs(E0)
    fx = np.fft.fftshift(np.fft.fftfreq(n_pts, d=dx))
    Kx, Ky = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fx, indexing='xy')
    E0_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E0)))
    k = 2.0 * np.pi / Ld
    kx2ky2 = Kx**2 + Ky**2
    kz = np.sqrt(np.maximum(0.0, k**2 - kx2ky2)) + 1j * np.sqrt(np.maximum(0.0, kx2ky2 - k**2))
    H = np.exp(1j * kz * z)
    Ez_fft = E0_fft * H
    Ez = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ez_fft)))
    Gau_pro = np.abs(Ez)
    return Gau, Gau_pro
import math

def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    z_values = [float(val) for val in z]
    inv_R0 = 0.0 if (math.isinf(R0) or R0 == 0) else 1.0 / R0
    inv_q0 = inv_R0 - 1j * (Mp2 * wavelength) / (math.pi * w0 ** 2)
    q0 = 1.0 / inv_q0
    q_lens_in = q0 + s
    A, B = Mf1[0][0], Mf1[0][1]
    C, D = Mf1[1][0], Mf1[1][1]
    q_lens_out = (A * q_lens_in + B) / (C * q_lens_in + D)
    q_ref = q_lens_out + L1
    wz = []
    for zi in z_values:
        q_z = q_ref + zi
        inv_q_z = 1.0 / q_z
        im_inv_q = inv_q_z.imag
        if im_inv_q >= 0:
            im_inv_q = -1e-30
        wz.append(math.sqrt(-wavelength / (math.pi * im_inv_q)))
    return wz
def Gussian_Lens_transmission(N, Ld, z, L, w0, R0, Mf1, Mp2, L1, s):
    """
    Simulate Gaussian-beam transmission through free space and a lens system,
    determine the new focal position, and return both the on-axis waist
    evolution and the transverse intensity distribution at that focus.
    """
    z_axis = np.asarray(z, dtype=float)
    Wz = np.array(
        gaussian_beam_through_lens(
            wavelength=Ld,
            w0=w0,
            R0=R0,
            Mf1=Mf1,
            z=z_axis,
            Mp2=Mp2,
            L1=L1,
            s=s,
        )
    )
    idx_focus = int(np.argmin(Wz))
    focus_depth = float(z_axis[idx_focus])
    total_distance = s + L1 + focus_depth
    _, field_focus = propagate_gaussian_beam(
        N=N,
        Ld=Ld,
        w0=w0,
        z=total_distance,
        L=L,
    )
    Intensity = np.abs(field_focus) ** 2
    return Wz, focus_depth, Intensity

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('28.3', 4)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
Ld = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 150  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 0
# User input for beam quality index
Mp2 = 1
N = 800  
L = 16*10**-3
assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
Ld= 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 100 # initial waist position
# User input for beam quality index
Mp2 = 1.5
N = 800  
L = 16*10**-3
assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
Ld = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 150 # initial waist position
# User input for beam quality index
Mp2 = 1.5
N = 800  
L = 16*10**-3
assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)
target = targets[3]

Ld = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 150 # initial waist position
# User input for beam quality index
Mp2 = 1.5
N = 800  
L = 16*10**-3
Wz, _, _ = Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s)
scalingfactor = max(z)/len(z)
LensWz = Wz[round(L1/scalingfactor)]
BeforeLensWz = Wz[round((L1-5)/scalingfactor)]
AfterLensWz = Wz[round((L1+5)/scalingfactor)]
assert (LensWz>BeforeLensWz, LensWz>AfterLensWz) == target
