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
