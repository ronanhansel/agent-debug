import numpy as np
from scipy.integrate import simps
# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson


def simulate_light_diffraction(n, d, RL, R0, lambda_):
    mr2 = 51
    ne2 = 61
    mr0 = 81
    Rout = 4 * R0
    r0 = np.linspace(0, R0, mr0)
    e0 = np.linspace(0, 2 * np.pi, ne2, endpoint=False)
    r2 = np.linspace(0, Rout, mr2)
    e2 = np.linspace(0, 2 * np.pi, ne2, endpoint=False)
    rr0, ee0 = np.meshgrid(r0, e0, indexing='ij')
    Ein = np.exp(- (rr0 ** 2) / (R0 ** 2))
    RR = RL**2 - rr0**2
    RR[RR < 0] = 0
    delta_L = (n-1) * (2 * RL - 2 * np.sqrt(RR))
    phi_lens = -2 * np.pi / lambda_ * delta_L
    Ein_lens = Ein * np.exp(1j * phi_lens)
    f = RL / (2 * (n - 1))
    Eout = np.zeros((mr2, ne2), dtype=np.complex128)
    dr0 = R0 / (mr0 - 1)
    de0 = 2 * np.pi / ne2
    for ir2, r_2 in enumerate(r2):
        for ie2, e_2 in enumerate(e2):
            phase = -2 * np.pi * rr0 * r_2 * np.cos(ee0 - e_2) / (lambda_ * f)
            integrand = Ein_lens * np.exp(1j * phase) * rr0
            Eout[ir2, ie2] = np.sum(integrand) * dr0 * de0
    Ie = np.abs(Eout)**2
    if Ie.max() > 0:
        Ie = Ie / Ie.max()
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
