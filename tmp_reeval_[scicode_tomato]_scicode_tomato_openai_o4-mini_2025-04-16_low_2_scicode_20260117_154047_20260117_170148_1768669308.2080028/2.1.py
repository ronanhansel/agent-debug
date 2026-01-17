# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
from scipy.integrate import simps

def simulate_light_diffraction(n, d, RL, R0, lambda_):
    '''Function to simulate light diffraction through a symmetric spherical lens.
    Inputs:
        n (float): refractive index of the lens material
        d (float): center thickness of the lens in mm (not needed for intensity)
        RL (float): radius of curvature of lens surfaces in mm (convex positive)
        R0 (float): radius of the incident Gaussian beam in mm
        lambda_ (float): wavelength of the incident light in mm
    Outputs:
        Ie (numpy.ndarray): 2D intensity array of shape (mr2, ne2)
    '''
    if R0 <= 0 or lambda_ <= 0:
        raise ValueError("Beam radius and wavelength must be positive.")
    if abs(RL) < R0:
        raise ValueError(f"Beam radius R0={R0} exceeds lens curvature radius RL={RL}.")
    k = 2 * np.pi / lambda_
    mr0, mr2, ne2 = 81, 51, 61
    r0 = np.linspace(0.0, R0, mr0)
    theta0 = np.linspace(0.0, 2.0 * np.pi, ne2, endpoint=False)
    sag = RL - np.sqrt(np.maximum(RL**2 - r0**2, 0.0))
    delta_d = -2.0 * sag
    E0 = np.exp(-(r0 / R0)**2)
    T = np.exp(-1j * k * (n - 1.0) * delta_d)
    E_pupil = (E0 * T)[:, None]
    f = RL / (n - 1.0)
    z = f
    r2 = np.linspace(0.0, R0, mr2)
    theta2 = np.linspace(0.0, 2.0 * np.pi, ne2, endpoint=False)
    E_out = np.zeros((mr2, ne2), dtype=complex)
    for i, rr in enumerate(r2):
        rr2 = rr**2
        for j, th2 in enumerate(theta2):
            dth = th2 - theta0
            cosd = np.cos(dth)[None, :]
            r0_mat = r0[:, None]
            phase = np.exp(1j * k / (2.0 * z) * (rr2 + r0_mat**2 - 2.0 * rr * r0_mat * cosd))
            integrand = E_pupil * phase * r0_mat
            I_theta = simps(integrand, theta0, axis=1)
            E_out[i, j] = simps(I_theta, r0)
    Ie = np.abs(E_out)**2
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
