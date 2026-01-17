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
def tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax):
    """
    Integrate the Tolman–Oppenheimer–Volkoff equations for a neutron star with
    a polytropic equation of state and return its gravitational mass and the
    gravitational time-dilation (lapse) at the centre.

    Parameters
    ----------
    rhoc : float
        Central rest-mass density ρ_c  ( > 0 ).
    eos_Gamma : float
        Polytropic exponent Γ         ( > 1 ).
    eos_kappa : float
        Polytropic coefficient κ       ( > 0 ).
    npoints : int
        Number of radial grid points   ( ≥ 2 ).
    rmax : float
        Outer radius of the numerical grid; must exceed the stellar surface.

    Returns
    -------
    (star_mass, star_lapse) : tuple(float, float)
        star_mass  – total gravitational mass M of the star.
        star_lapse – lapse α = e^{φ_c} giving time-dilation at the centre.
    """
    if rhoc <= 0.0:
        raise ValueError("Central density 'rhoc' must be positive.")
    if eos_Gamma <= 1.0:
        raise ValueError("Adiabatic exponent 'eos_Gamma' must be greater than 1.")
    if eos_kappa <= 0.0:
        raise ValueError("Polytropic coefficient 'eos_kappa' must be positive.")
    if npoints < 2:
        raise ValueError("'npoints' must be at least 2.")
    if rmax <= 0.0:
        raise ValueError("'rmax' must be positive.")
    r = np.linspace(0.0, rmax, npoints, dtype=float)
    dr = r[1] - r[0]
    Pc = eos_press_from_rho(rhoc, eos_Gamma, eos_kappa)
    u = np.zeros((3, npoints), dtype=float)
    u[:, 0] = (Pc, 0.0, 0.0)
    surface_index = None
    for i in range(1, npoints):
        r_prev = r[i - 1]
        y_prev = u[:, i - 1]
        if y_prev[0] <= 0.0:
            u[:, i:] = y_prev[:, None]
            surface_index = i - 1
            break
        k1 = np.array(tov_RHS(tuple(y_prev), r_prev, eos_Gamma, eos_kappa))
        k2 = np.array(tov_RHS(tuple(y_prev + 0.5 * dr * k1), r_prev + 0.5 * dr, eos_Gamma, eos_kappa))
        k3 = np.array(tov_RHS(tuple(y_prev + 0.5 * dr * k2), r_prev + 0.5 * dr, eos_Gamma, eos_kappa))
        k4 = np.array(tov_RHS(tuple(y_prev + dr * k3), r_prev + dr, eos_Gamma, eos_kappa))
        y_next = y_prev + (dr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_next[0] = max(0.0, y_next[0])
        u[:, i] = y_next
        if y_next[0] == 0.0:
            surface_index = i
            break
    if surface_index is None:
        surface_index = npoints - 1
    R_star = r[surface_index]
    star_mass = float(u[1, surface_index])
    if R_star == 0.0 or (2.0 * star_mass / R_star) >= 1.0:
        raise RuntimeError("Invalid configuration: R=0 or 2M/R ≥ 1.")
    phi_int_surface = u[2, surface_index]
    phi_ext_surface = 0.5 * np.log(1.0 - 2.0 * star_mass / R_star)
    phi_shift = phi_ext_surface - phi_int_surface
    star_lapse = float(np.exp(u[2, 0] + phi_shift))
    return star_mass, star_lapse

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('58.5', 4)
target = targets[0]

rhoc = 0.3
eos_Gamma = 2.1
eos_kappa = 30
npoints = 200
rmax = 100000.
assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)
target = targets[1]

rhoc = 2e-5
eos_Gamma = 1.8
eos_kappa = 20
npoints = 2000
rmax = 100.
assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)
target = targets[2]

rhoc = 1.28e-3
eos_Gamma = 5./3.
eos_kappa = 80.
npoints = 200000
rmax = 100.
assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)
target = targets[3]

rhoc = 1.28e-3
# equation of state
eos_Gamma = 2.0
eos_kappa = 100.
# grid for integration
rmax = 100.
npoints = 200000
assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)
