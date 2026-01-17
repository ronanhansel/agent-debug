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
def tov_RHS(data, r, eos_Gamma, eos_kappa):
    '''
    Compute the integrand (radial derivatives) of the Tolman-Oppenheimer-Volkoff
    equations for a spherically symmetric neutron star described by a
    polytropic equation of state.

    Inputs
    ------
    data : tuple(float, float, float)
        Current values of (press, mass, phi) at radius r.
    r : float
        Radius at which to evaluate the RHS (≥ 0).
    eos_Gamma : float
        Adiabatic exponent Γ of the EOS.
    eos_kappa : float
        Polytropic coefficient κ of the EOS.

    Outputs
    -------
    rhs : tuple(float, float, float)
        Radial derivatives (dP/dr, dM/dr, dφ/dr).  Outside the star
        (press ≤ 0) and at the centre (r = 0) all derivatives are zero.
    '''
    press, mass, phi = data
    if press <= 0.0 or r == 0.0:
        return (0.0, 0.0, 0.0)
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)
    eps = eos_eps_from_press(press, eos_Gamma, eos_kappa)
    energy_density = rho * (1.0 + eps)
    denom = r * (r - 2.0 * mass)
    if abs(denom) < 1e-12:
        return (0.0, 0.0, 0.0)
    pi = 3.141592653589793
    common = (mass + 4.0 * pi * r**3 * press) / denom
    dP_dr = - (energy_density + press) * common
    dM_dr = 4.0 * pi * r**2 * energy_density
    dphi_dr = common
    return (float(dP_dr), float(dM_dr), float(dphi_dr))

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
