import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    press = eos_kappa * rho**eos_Gamma
    return press
def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    """
    Compute density for a polytropic equation of state given the pressure.

    The polytropic relation is:
        press = eos_kappa * rho**eos_Gamma

    Inverted to give:
        rho = (press / eos_kappa)**(1.0 / eos_Gamma)

    Parameters
    ----------
    press : float
        Pressure value.
    eos_Gamma : float
        Adiabatic exponent of the equation of state.
    eos_kappa : float
        Polytropic coefficient.

    Returns
    -------
    rho : float
        Density corresponding to the given pressure.
    """
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    return rho
def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    """
    Compute specific internal energy for a polytropic equation of state given the pressure.
    
    Combines the polytropic EOS:
        press = eos_kappa * rho**eos_Gamma
    with the Gamma-law EOS:
        press = (eos_Gamma - 1) * rho * eps

    Parameters
    ----------
    press : float
        Pressure value.
    eos_Gamma : float
        Adiabatic exponent of the equation of state.
    eos_kappa : float
        Polytropic coefficient.

    Returns
    -------
    eps : float
        Specific internal energy corresponding to the given pressure.
    """
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    eps = press / ((eos_Gamma - 1.0) * rho)
    return eps
def tov_RHS(data, r, eos_Gamma, eos_kappa):
    """
    Compute the right‐hand‐side integrand of the Tolman–Oppenheimer–Volkoff equations
    for a polytropic equation of state.

    Inputs:
      data       : tuple of floats (press, mass, phi)
      r          : float, current radius
      eos_Gamma  : float, adiabatic exponent
      eos_kappa  : float, polytropic coefficient

    Returns:
      tuple of floats (dpress_dr, dmass_dr, dphi_dr).
      If r == 0 or press <= 0, returns (0.0, 0.0, 0.0).
    """
    press, mass, phi = data
    if r == 0.0 or press <= 0.0:
        return 0.0, 0.0, 0.0
    rho = rho_from_press(press, eos_Gamma, eos_kappa)
    denom     = r * (r - 2.0 * mass)
    dmass_dr  = 4.0 * np.pi * r**2 * rho
    common    = (mass + 4.0 * np.pi * r**3 * press) / denom
    dpress_dr = - (rho + press) * common
    dphi_dr   =   common
    return dpress_dr, dmass_dr, dphi_dr

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
