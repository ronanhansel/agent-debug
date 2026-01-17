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
def tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax):
    """
    Compute gravitational mass and central time dilation (lapse) of a TOV star.

    Inputs:
      rhoc        = central density
      eos_Gamma   = adiabatic exponent Γ
      eos_kappa   = polytropic constant κ
      npoints     = number of radial grid points
      rmax        = maximum radius to integrate to

    Outputs:
      star_mass   = gravitational mass of the star (G=c=Msun=1)
      star_lapse  = gravitational time dilation at the center
    """
    p_c = eos_press_from_rho(rhoc, eos_Gamma, eos_kappa)
    y0 = [p_c, 0.0, 0.0]
    r = np.linspace(0.0, rmax, npoints)
    sol = si.odeint(tov_RHS, y0, r, args=(eos_Gamma, eos_kappa))
    press = sol[:, 0]
    mass = sol[:, 1]
    phi = sol[:, 2]
    zero_idx = np.where(press <= 0.0)[0]
    i_surf = zero_idx[0] if zero_idx.size > 0 else (npoints - 1)
    R_star = r[i_surf]
    M_star = mass[i_surf]
    phi_int = phi[i_surf]
    phi_ext = 0.5 * np.log(1.0 - 2.0 * M_star / R_star)
    star_lapse = np.exp(phi_ext - phi_int)
    return M_star, star_lapse

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
