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
