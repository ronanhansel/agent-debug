# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
from scipy.integrate import simps

def simulate_light_diffraction(n, d, RL, R0, lambda_):
    # Custom linspace to avoid external dependencies
    def _linspace(start, stop, num, endpoint=True):
        if num <= 0:
            return []
        if num == 1:
            return [start]
        if endpoint:
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
        else:
            step = (stop - start) / num
            return [start + i * step for i in range(num)]
    pi = 3.141592653589793
    e_const = 2.718281828459045
    if RL == 0.0:
        raise ValueError("Lens curvature radius RL must be non-zero.")
    inv_f = (n - 1.0) * (2.0 / RL - (n - 1.0) * d / (n * RL ** 2))
    if inv_f == 0.0:
        raise ValueError("Computed focal length is infinite (1/f = 0).")
    f = 1.0 / inv_f
    w = (lambda_ * f) / (pi * R0 * (2.0 ** 0.5))
    mr2, ne2 = 51, 61
    r_max = 3.0 * w
    r = _linspace(0.0, r_max, mr2, endpoint=True)
    theta = _linspace(0.0, 2.0 * pi, ne2, endpoint=False)  #
    coeff = -2.0 * (pi ** 2) * (R0 ** 2) / (lambda_ ** 2 * f ** 2)
    I_r = [pow(e_const, coeff * (ri ** 2)) for ri in r]
    max_I = max(I_r) if I_r else 1.0
    Ie = [[val / max_I for _ in theta] for val in I_r]
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
