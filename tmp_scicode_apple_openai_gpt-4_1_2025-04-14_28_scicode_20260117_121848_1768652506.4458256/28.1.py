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
