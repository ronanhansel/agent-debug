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
def tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax):
    '''Compute the gravitational mass and central time dilation (lapse)
    of a polytropic TOV star with central density rhoc.'''
    if rhoc <= 0.0:
        return 0.0, 1.0
    press_c = eos_press_from_rho(rhoc, eos_Gamma, eos_kappa)
    r = np.linspace(0.0, rmax, npoints)
    y0 = [press_c, 0.0, 0.0]
    sol = si.solve_ivp(
        fun=lambda rr, yy: tov_RHS((yy[0], yy[1], yy[2]), rr, eos_Gamma, eos_kappa),
        t_span=(r[0], r[-1]),
        y0=y0,
        t_eval=r,
        vectorized=False,
        atol=1e-8,
        rtol=1e-8
    )
    press, mass, phi = sol.y
    zero_idxs = np.where(press <= 0.0)[0]
    if zero_idxs.size > 0:
        i_surf = max(zero_idxs[0] - 1, 0)
    else:
        i_surf = len(r) - 1
    star_mass = mass[i_surf]
    star_radius = r[i_surf]
    phi_int = phi[i_surf]
    if star_radius <= 0.0:
        return star_mass, 1.0
    phi_ext = 0.5 * np.log(1.0 - 2.0 * star_mass / star_radius)
    delta_phi = phi_ext - phi_int
    star_lapse = np.exp(phi[0] + delta_phi)
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
