import numpy as np
from scipy.integrate import simpson

def propagate_gaussian_beam(N, Ld, w0, z, L):
    '''Propagate a Gaussian beam and calculate its intensity distribution before and after propagation.

    Parameters
    ----------
    N : int
        Number of sampling points in each dimension (assumes a square grid).
    Ld : float
        Wavelength of the Gaussian beam.
    w0 : float
        Waist radius of the Gaussian beam at its narrowest point.
    z : float
        Propagation distance of the Gaussian beam.
    L : float
        Side length of the square area over which the beam is sampled.

    Returns
    -------
    Gau : ndarray, shape (N+1, N+1)
        Magnitude of the beam’s complex amplitude at z = 0 (initial field).
    Gau_pro : ndarray, shape (N+1, N+1)
        Magnitude of the beam’s complex amplitude after propagating a distance z.
    '''
    dx = L / N
    x = np.linspace(-L / 2.0, L / 2.0, N + 1)
    X, Y = np.meshgrid(x, x, indexing='xy')
    E0 = np.exp(-(X**2 + Y**2) / w0**2)
    Gau = np.abs(E0)
    fx = np.fft.fftshift(np.fft.fftfreq(N + 1, d=dx))
    FX, FY = np.meshgrid(fx, fx, indexing='xy')
    H = np.exp(-1j * np.pi * Ld * z * (FX**2 + FY**2))
    E0_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E0)))
    Ez_ft = E0_ft * H
    Ez = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ez_ft)))
    Gau_pro = np.abs(Ez)
    return Gau, Gau_pro
def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    """
    Calculate the Gaussian-beam radius w(z) after transmission through a lens.
    """

    pi = 3.141592653589793
    eps = 2.220446049250313e-16

    def _is_inf(value):
        return value == float("inf") or value == float("-inf")

    def q_from_wR(w, R):
        inv_R = 0.0 if _is_inf(R) or R == 0 else 1.0 / R
        inv_q = inv_R - 1j * Mp2 * wavelength / (pi * w * w)
        return 1.0 / inv_q

    def apply_abcd(q_in, M):
        A, B = M[0][0], M[0][1]
        C, D = M[1][0], M[1][1]
        return (A * q_in + B) / (C * q_in + D)

    q_source = q_from_wR(w0, R0)
    q_at_lens_in = q_source + s
    q_after_lens = apply_abcd(q_at_lens_in, Mf1)
    q_ref = q_after_lens + L1

    try:
        z_iter = iter(z)
        z_values = list(z_iter)
    except TypeError:
        z_values = [z]

    results = []
    for zi in z_values:
        q_z = q_ref + zi
        inv_q = 1.0 / q_z
        im_inv_q = inv_q.imag
        if im_inv_q >= 0:
            im_inv_q = -eps
        w_z = ((-wavelength) / (pi * im_inv_q)) ** 0.5
        results.append(w_z)

    return results[0] if len(results) == 1 else results
def Gussian_Lens_transmission(
    N: int,
    Ld: float,
    z,
    L: float,
    w0: float,
    R0: float,
    Mf1,
    Mp2: float,
    L1: float,
    s: float
):
    '''Simulate a Gaussian-beam propagating through free space and a lens system.

    Parameters
    ----------
    N   : int
        Number of sampling points in each transverse dimension (grid is (N+1)×(N+1)).
    Ld  : float
        Wavelength of the Gaussian beam (metres).
    z   : array_like
        1-D distances (metres) after the reference plane following the lens where the beam
        radius is to be evaluated.
    L   : float
        Physical side length (metres) of the square computational window.
    w0  : float
        Initial 1/e radius (waist) of the Gaussian beam at the source plane (metres).
    R0  : float
        Initial radius of curvature of the beam’s wave-front (metres).  Use np.inf for planar.
    Mf1 : (2,2) array_like
        ABCD matrix of the lens (or first optical element).
    Mp2 : float
        M²-like scaling factor used in the complex-beam-parameter definition.
    L1  : float
        Distance (metres) from the lens to the reference plane after the lens.
    s   : float
        Distance (metres) from the initial waist plane to the lens.

    Returns
    -------
    Wz : ndarray, shape (len(z),)
        Beam radii evaluated at each axial position in `z`.
    focus_depth : float
        Position along the propagation axis (metres, from the reference plane) where the
        beam achieves its minimum waist (new focus).
    Intensity : ndarray, shape (N+1, N+1)
        Normalised (peak = 1) transverse intensity distribution in the focal plane.
    '''
    # Ensure z is a 1-D numpy array for uniform processing
    z = np.atleast_1d(np.asarray(z, dtype=float))

    # --- Compute beam radius evolution W(z) using the provided ABCD helper ----------------
    Wz_vals = gaussian_beam_through_lens(
        wavelength=Ld,
        w0=w0,
        R0=R0,
        Mf1=Mf1,
        z=z,
        Mp2=Mp2,
        L1=L1,
        s=s
    )
    Wz = np.asarray(Wz_vals, dtype=float)

    # --- Determine new focus position and waist size --------------------------------------
    min_idx = int(np.argmin(Wz))
    focus_depth = float(z[min_idx])    # z (after reference plane) of minimum waist
    w_focus = float(Wz[min_idx])       # radius at new focus

    # --- Build transverse grid covering side length L -------------------------------------
    coords = np.linspace(-L / 2.0, L / 2.0, N + 1)
    X, Y = np.meshgrid(coords, coords, indexing='xy')

    # --- Calculate intensity distribution at the focal plane ------------------------------
    # At the waist, the field is Gaussian with amplitude ∝ exp(−(x²+y²)/w_f²); intensity ∝ |E|²
    Intensity = np.exp(-2.0 * (X**2 + Y**2) / (w_focus ** 2))
    Intensity /= Intensity.max()  # normalise peak intensity to 1

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
