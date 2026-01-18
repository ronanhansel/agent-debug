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
