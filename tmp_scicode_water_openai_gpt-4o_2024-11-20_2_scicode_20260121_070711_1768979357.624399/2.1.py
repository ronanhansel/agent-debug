# Compatibility shims for deprecated NumPy/SciPy APIs
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
# Note: scipy.integrate.simps -> simpson shim handled at import level
# The benchmark uses 'from scipy.integrate import simps' so we alias in scipy.integrate module
try:
    from scipy import integrate as _sci_int
    if not hasattr(_sci_int, 'simps'):
        _sci_int.simps = _sci_int.simpson
except ImportError:
    pass

import numpy as np
from scipy.integrate import simpson as simps

def simulate_light_diffraction(n, d, RL, R0, lambda_):
    def gaussian_beam_amplitude(R, w0):
        return np.exp(-R**2 / w0**2)
    
    def lens_phase_shift(R, d, RL, k, n):
        d_eff = d - (R**2 / (2 * RL))
        return -k * (n - 1) * d_eff

    mr2 = 51
    ne2 = 61
    mr0 = 81
    
    r = np.linspace(0, R0, mr0)
    theta = np.linspace(0, 2 * np.pi, ne2)
    R, Theta = np.meshgrid(r, theta)
    
    w0 = R0
    A = gaussian_beam_amplitude(R, w0)
    
    k = 2 * np.pi / lambda_
    phi = lens_phase_shift(R, d, RL, k, n)
    
    U = A * np.exp(1j * phi)
    
    x = np.linspace(-R0, R0, mr2)
    y = np.linspace(-R0, R0, mr2)
    X, Y = np.meshgrid(x, y)
    
    U_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))
    intensity_fft = np.abs(U_fft)**2
    
    Ie = intensity_fft / np.max(intensity_fft)
    
    return Ie

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('2.1', 4)
target = targets[0]

n=1.5062
d=3
RL=0.025e3
R0=1
lambda_=1.064e-3
assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
target = targets[1]

n=1.5062
d=4
RL=0.05e3
R0=1
lambda_=1.064e-3
assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
target = targets[2]

n=1.5062
d=2
RL=0.05e3
R0=1
lambda_=1.064e-3
assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)
target = targets[3]

n=1.5062
d=2
RL=0.05e3
R0=1
lambda_=1.064e-3
intensity_matrix = simulate_light_diffraction(n, d, RL, R0, lambda_)
max_index_flat = np.argmax(intensity_matrix)
assert (max_index_flat==0) == target
