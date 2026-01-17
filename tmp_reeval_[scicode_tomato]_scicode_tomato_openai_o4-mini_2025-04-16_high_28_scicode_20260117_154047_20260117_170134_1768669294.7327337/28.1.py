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
