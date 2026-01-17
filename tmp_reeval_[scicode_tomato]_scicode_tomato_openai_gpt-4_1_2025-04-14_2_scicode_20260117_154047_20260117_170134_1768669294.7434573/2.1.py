# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import numpy as np
from scipy.integrate import simps

def simulate_light_diffraction(n, d, RL, R0, lambda_):
    mr2, ne2 = 51, 61
    mr0 = 81
    pi = 3.141592653589793

    f = RL / (2 * (n - 1))

    r0 = [i * R0 / (mr0 - 1) for i in range(mr0)]
    dr0 = r0[1] - r0[0]
    theta0 = [i * 2 * pi / ne2 for i in range(ne2)]
    dtheta0 = theta0[1] - theta0[0]
    r0g = [[r0[i] for _ in range(ne2)] for i in range(mr0)]
    th0g = [[theta0[j] for j in range(ne2)] for _ in range(mr0)]
    x0 = [[r0g[i][j] * __import__('math').cos(th0g[i][j]) for j in range(ne2)] for i in range(mr0)]
    y0 = [[r0g[i][j] * __import__('math').sin(th0g[i][j]) for j in range(ne2)] for i in range(mr0)]

    import cmath
    E0 = [[__import__('math').exp(- (r0g[i][j] / R0) ** 2) for j in range(ne2)] for i in range(mr0)]

    k = 2 * pi / lambda_
    lens_phase = [[cmath.exp(-1j * k * r0g[i][j]**2 / (2 * f)) for j in range(ne2)] for i in range(mr0)]

    Uin = [[E0[i][j] * lens_phase[i][j] for j in range(ne2)] for i in range(mr0)]

    r_max = 2 * R0
    r_obs = [i * r_max / (mr2 - 1) for i in range(mr2)]
    theta_obs = [i * 2 * pi / ne2 for i in range(ne2)]
    r_obs_g = [[r_obs[i] for _ in range(ne2)] for i in range(mr2)]
    theta_obs_g = [[theta_obs[j] for j in range(ne2)] for _ in range(mr2)]
    x_obs = [[r_obs_g[i][j] * __import__('math').cos(theta_obs_g[i][j]) for j in range(ne2)] for i in range(mr2)]
    y_obs = [[r_obs_g[i][j] * __import__('math').sin(theta_obs_g[i][j]) for j in range(ne2)] for i in range(mr2)]

    area = [[r0g[i][j] * dr0 * dtheta0 for j in range(ne2)] for i in range(mr0)]

    Ie = [[0.0 for _ in range(ne2)] for _ in range(mr2)]

    for i in range(mr2):
        for j in range(ne2):
            U_out = 0+0j
            for p in range(mr0):
                for q in range(ne2):
                    phase = cmath.exp(-1j * k * (x_obs[i][j] * x0[p][q] + y_obs[i][j] * y0[p][q]) / f)
                    U_out += Uin[p][q] * phase * area[p][q]
            Ie[i][j] = abs(U_out) ** 2

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
