import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    '''
    Propagate a Gaussian beam and calculate its intensity distribution before and after propagation.
    Input
    N : int
        The number of sampling points in each dimension (assumes a square grid,
        total points per axis = N+1).
    Ld : float
        Wavelength of the Gaussian beam.
    w0 : float
        Waist radius of the Gaussian beam at its narrowest point.
    z : float
        Propagation distance of the Gaussian beam.
    L : float
        Side length of the square area over which the beam is sampled.
    Output
    Gau     : 2D array, shape (N+1, N+1), absolute amplitude |E(x,y,0)| before propagation.
    Gau_pro : 2D array, shape (N+1, N+1), absolute amplitude |E(x,y,z)| after propagation.
    '''
    M = N + 1
    dx = L / N
    x = np.linspace(-L/2, L/2, M)
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X**2 + Y**2) / w0**2)
    E0_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E0))) * (dx**2)
    fx = np.fft.fftshift(np.fft.fftfreq(M, d=dx))
    FX, FY = np.meshgrid(fx, fx)
    k = 2 * np.pi / Ld
    arg = 1.0 - (Ld * FX)**2 - (Ld * FY)**2
    H = np.exp(1j * k * z * np.sqrt(arg + 0j))
    E_ft_prop = E0_ft * H
    E_prop = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_ft_prop))) / (dx**2)
    Gau = np.abs(E0)
    Gau_pro = np.abs(E_prop)
    return Gau, Gau_pro
def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    """
    gaussian_beam_through_lens simulates the propagation of a Gaussian beam through an optical lens system
    and calculates the beam waist size at various distances from the second optical element.
    Inputs:
      - wavelength: float, λ (m)
      - w0: float, initial beam waist radius at the source (m)
      - R0: float, initial wavefront radius of curvature at the source (m)
      - Mf1: 2×2 ndarray, ABCD matrix of the first lens/element
      - z: 1D ndarray of floats, distances (m) from the second optical element
      - Mp2: either a 2×2 ndarray (ABCD matrix of 2nd element) or a scalar focal length f₂ (float)
      - L1: float, distance (m) from first element to second element
      - s: float, distance (m) from source waist to first element
    Output:
      - wz: 1D ndarray of floats, beam waist radius at each z
    """
    q0 = 1.0 / (1.0 / R0 - 1j * (wavelength / (np.pi * w0**2)))
    M_s = np.array([[1.0, s],
                    [0.0, 1.0]], dtype=complex)
    M_f1 = Mf1.astype(complex)
    M1 = M_f1 @ M_s
    M_L1 = np.array([[1.0, L1],
                     [0.0, 1.0]], dtype=complex)
    if np.isscalar(Mp2):
        f2 = Mp2
        M_p2 = np.array([[1.0,    0.0],
                         [-1.0/f2, 1.0]], dtype=complex)
    else:
        M_p2 = Mp2.astype(complex)
    M2 = M_p2 @ M_L1 @ M1
    wz = np.zeros_like(z, dtype=float)
    for idx, zi in enumerate(z):
        M_z = np.array([[1.0, zi],
                        [0.0, 1.0]], dtype=complex)
        M_tot = M_z @ M2
        A, B = M_tot[0,0], M_tot[0,1]
        C, D = M_tot[1,0], M_tot[1,1]
        qz = (A * q0 + B) / (C * q0 + D)
        wz[idx] = np.sqrt(-wavelength / (np.pi * np.imag(1.0 / qz)))
    return wz
def Gussian_Lens_transmission(N, Ld, z, L, w0, R0, Mf1, Mp2, L1, s):
    Wz = gaussian_beam_through_lens(
        wavelength=Ld,
        w0=w0,
        R0=R0,
        Mf1=Mf1,
        z=z,
        Mp2=Mp2,
        L1=L1,
        s=s
    )
    idx_min = np.argmin(Wz)
    focus_depth = z[idx_min]
    w_focus = Wz[idx_min]
    _, amplitude = propagate_gaussian_beam(
        N=N,
        Ld=Ld,
        w0=w_focus,
        z=focus_depth,
        L=L
    )
    Intensity = amplitude**2
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
