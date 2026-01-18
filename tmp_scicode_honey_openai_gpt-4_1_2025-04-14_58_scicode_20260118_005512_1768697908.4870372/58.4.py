import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    press = eos_kappa * rho ** eos_Gamma
    return press
def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    return rho
def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    eps = press / ((eos_Gamma - 1.0) * rho)
    return eps
def tov_RHS(data, r, eos_Gamma, eos_kappa):
    press, mass, phi = data
    if press <= 0.0:
        return (0.0, 0.0, 0.0)
    if r == 0.0:
        return (0.0, 0.0, 0.0)
    def rho_from_press(press, eos_Gamma, eos_kappa):
        return (press / eos_kappa) ** (1.0 / eos_Gamma)
    def eps_from_press(press, eos_Gamma, eos_kappa):
        rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
        return press / ((eos_Gamma - 1.0) * rho)
    rho = rho_from_press(press, eos_Gamma, eos_kappa)
    eps = eps_from_press(press, eos_Gamma, eos_kappa)
    pi = 3.141592653589793
    denominator = r * (r - 2 * mass)
    if denominator == 0.0:
        dpress = 0.0
        dphi = 0.0
    else:
        numer_common = mass + 4 * pi * r ** 3 * press
        dpress = - (rho + press + rho * eps) * numer_common / denominator
        dphi = numer_common / denominator
    dmass = 4 * pi * r ** 2 * rho
    return (dpress, dmass, dphi)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('58.4', 3)
target = targets[0]

data = (1e35, 0.0, 0.0)  # High pressure, mass = 0, phi = 0 at the origin
r = 0.0
eos_Gamma = 2.0
eos_kappa = 1e-10
assert np.allclose(tov_RHS(data, r, eos_Gamma, eos_kappa), target)
target = targets[1]

data = (10, 20, 1.0)  # Moderate pressure, some mass, some phi inside the star
r = 1e3
eos_Gamma = 2.0
eos_kappa = 1e-3
assert np.allclose(tov_RHS(data, r, eos_Gamma, eos_kappa), target)
target = targets[2]

data = (0.3, 1e3, 1.0)
r = 20
eos_Gamma = 2
eos_kappa = 100
assert np.allclose(tov_RHS(data, r, eos_Gamma, eos_kappa), target)
