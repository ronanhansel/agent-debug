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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('28.1', 3)
target = targets[0]

N = 500  # sample number ,
L = 10*10**-3   # Full side length 
Ld = 0.6328 * 10**-6  
w0 = 1.0 * 10**-3  
z = 10  
gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
# domain specific check 1
# Physics Calculate the energy info at beginning and end 
# The result is Intensity distrbution
# The total energy value is expressed as total power, the intensity integrated over the cross section 
dx = L/N
dy = L/N
dA = dx*dy
P1 = np.sum(gau1) * dA
P2 = np.sum(gau2)* dA
assert np.allclose((P1, P2), target)
target = targets[1]

N = 800  
L = 16*10**-3
Ld = 0.6328 * 10**-6  
w0 = 1.5* 10**-3  
z = 15  
gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
# domain specific check 2
# Physics Calculate the energy info at beginning and end 
# The result is Intensity distrbution
# The total energy value is expressed as total power, the intensity integrated over the cross section 
dx = L/N
dy = L/N
dA = dx*dy
P1 = np.sum(gau1) * dA
P2 = np.sum(gau2)* dA
assert np.allclose((P1, P2), target)
target = targets[2]

N = 400 
L = 8*10**-3
Ld = 0.6328 * 10**-6  
w0 = 1.5* 10**-3  
z = 20  
gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
# domain specific 3
# Physics Calculate the energy info at beginning and end 
# The result is Intensity distrbution
# The total energy value is expressed as total power, the intensity integrated over the cross section 
dx = L/N
dy = L/N
dA = dx*dy
P1 = np.sum(gau1) * dA
P2 = np.sum(gau2)* dA
assert np.allclose((P1, P2), target)
