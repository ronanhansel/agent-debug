import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    press = eos_kappa * rho**eos_Gamma
    return press
def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    return rho
def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    """
    This function computes specific internal energy for a polytropic equation of state given the pressure.
    Inputs:
    press: the pressure, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float.
    eos_kappa: coefficient of the equation of state, a float.
    Outputs:
    eps: the specific internal energy, a float.
    """
    eps = press / ((eos_Gamma - 1.0) * (press / eos_kappa)**(1.0 / eos_Gamma))
    return eps
def tov_RHS(data, r, eos_Gamma, eos_kappa):
    press, mass, phi = data
    if r <= 0.0 or press <= 0.0:
        return (0.0, 0.0, 0.0)
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)
    eps = eos_eps_from_press(press, eos_Gamma, eos_kappa)
    G = 6.67430e-11
    c = 299792458
    pi = 3.141592653589793
    rho_total = rho * (1.0 + eps / c**2)
    denom = r * (r - 2.0 * G * mass / c**2)
    dpress_dr = - (rho_total + press / c**2) * (mass + 4.0 * pi * r**3 * press / c**2) / denom
    dmass_dr = 4.0 * pi * r**2 * rho
    dphi_dr = 2.0 * G * (mass + 4.0 * pi * r**3 * press / c**2) / (c**2 * denom)
    return (dpress_dr, dmass_dr, dphi_dr)

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
