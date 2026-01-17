# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
from scipy.integrate import simps

def simulate_light_diffraction(n, d, RL, R0, lambda_):
    '''Function to simulate light diffraction through a lens and compute the intensity distribution.
    Inputs:
      n (float): Refractive index of the lens material.
      d (float): Center thickness of the lens (mm).
      RL (float): Radius of curvature of both lens surfaces (mm).
      R0 (float): Radius of the incident Gaussian beam (mm).
      lambda_ (float): Wavelength of the incident light (mm).
    Output:
      Ie (np.ndarray): 2D intensity array of shape (mr2, ne2).
    '''
    mr0, mr2 = 81, 51
    ne2, ne1 = 61, 60
    k = 2 * np.pi / lambda_
    r0 = np.linspace(0, R0, mr0)
    E0 = np.exp(- (r0 / R0)**2)
    sag = 2*(RL - np.sqrt(RL**2 - r0**2)) - d
    phi = k * (n - 1) * sag
    E_lens = E0 * np.exp(1j * phi)
    r2 = np.linspace(0, R0, mr2)
    theta2 = np.linspace(0, 2*np.pi, ne2, endpoint=False)
    R1, R2 = RL, -RL
    inv_f = (n - 1)*(1/R1 - 1/R2 + (n - 1)*d/(n*R1*R2))
    f = 1.0 / inv_f
    ang = np.linspace(0, 2*np.pi, ne1, endpoint=False)
    cos_ang = np.cos(ang)
    pref = np.exp(1j * k * f) / (1j * lambda_ * f)
    quad_in = np.exp(1j * k * r0**2 / (2 * f))
    E2_r = np.zeros(mr2, dtype=complex)
    for i, rv in enumerate(r2):
        arg = k * r0 * rv / f
        J0 = np.real(np.mean(np.exp(1j * np.outer(arg, cos_ang)), axis=1))
        integrand = E_lens * quad_in * J0 * r0
        integral = simps(integrand, r0)
        E2_r[i] = pref * np.exp(1j * k * rv**2 / (2 * f)) * integral
    E2_2D = np.tile(E2_r[:, None], (1, ne2))
    Ie = np.abs(E2_2D)**2
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
