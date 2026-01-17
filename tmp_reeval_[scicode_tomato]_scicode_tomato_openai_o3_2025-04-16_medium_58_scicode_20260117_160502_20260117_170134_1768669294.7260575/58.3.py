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
def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    """
    Compute the specific internal energy (eps) for a fluid obeying a polytropic
    equation of state combined with a Gamma–law (ideal–fluid) relation.

    Governing relations
    -------------------
    1. Polytropic EOS :  P = κ · ρ^Γ
    2. Gamma–law EOS :  P = (Γ − 1) · ρ · ε

    From these, the density is  ρ = (P / κ)^{1/Γ}
    and the specific internal energy is
        ε = P / [(Γ − 1) · ρ].

    Parameters
    ----------
    press : float
        Pressure (must be non-negative).
    eos_Gamma : float
        Adiabatic exponent Γ (must satisfy Γ > 1).
    eos_kappa : float
        Polytropic coefficient κ (must be positive).

    Returns
    -------
    float
        Specific internal energy ε corresponding to the given pressure.

    Raises
    ------
    ValueError
        If `press` is negative, or if `eos_Gamma` ≤ 1, or if `eos_kappa` ≤ 0.
    """
    if press < 0.0:
        raise ValueError("Pressure 'press' must be non-negative.")
    if eos_Gamma <= 1.0:
        raise ValueError("Adiabatic exponent 'eos_Gamma' must be greater than 1.")
    if eos_kappa <= 0.0:
        raise ValueError("Polytropic coefficient 'eos_kappa' must be positive.")
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    if rho == 0.0:
        return 0.0
    eps = press / ((eos_Gamma - 1.0) * rho)
    return float(eps)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('58.3', 3)
target = targets[0]

press = 10
eos_Gamma = 15
eos_kappa = 20
assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)
target = targets[1]

press = 10000
eos_Gamma = 3./5.
eos_kappa = 80
assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)
target = targets[2]

press = 100
eos_Gamma = 2.
eos_kappa = 100.
assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)
