import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    """
    Compute pressure for a polytropic equation of state given the density.

    Parameters
    ----------
    rho : float
        Mass density (must be non-negative).
    eos_Gamma : float
        Adiabatic exponent (Γ) of the equation of state.
    eos_kappa : float
        Polytropic coefficient (κ) of the equation of state.

    Returns
    -------
    float
        Pressure corresponding to the given density.

    Raises
    ------
    ValueError
        If `rho` is negative.
    """
    if rho < 0:
        raise ValueError("Density 'rho' must be non-negative.")
    press = eos_kappa * (rho ** eos_Gamma)
    return float(press)
def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    """
    Compute density for a polytropic equation of state given the pressure.

    Parameters
    ----------
    press : float
        Pressure (must be non-negative).
    eos_Gamma : float
        Adiabatic exponent (Γ) of the equation of state.
    eos_kappa : float
        Polytropic coefficient (κ) of the equation of state.

    Returns
    -------
    float
        Density (ρ) corresponding to the given pressure.

    Raises
    ------
    ValueError
        If `press` is negative.
    """
    if press < 0:
        raise ValueError("Pressure 'press' must be non-negative.")
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    return float(rho)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('58.2', 3)
target = targets[0]

press = 10
eos_Gamma = 20
eos_kappa = 30
assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)
target = targets[1]

press = 1000
eos_Gamma = 50
eos_kappa = 80
assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)
target = targets[2]

press = 20000
eos_Gamma = 2.
eos_kappa = 100.
assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)
