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
    if eos_Gamma <= 1.0 or eos_kappa <= 0.0:
        raise ValueError("Gamma must be >1 and kappa >0 for physical EOS.")
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
    eps = press / ((eos_Gamma - 1.0) * rho)
    return eps
def tov_RHS(data, r, eos_Gamma, eos_kappa):
    press, mass, phi = data
    if press <= 0.0 or r <= 0.0:
        return (0.0, 0.0, 0.0)
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)
    eps = eos_eps_from_press(press, eos_Gamma, eos_kappa)
    pi = 3.141592653589793
    dmass_dr = 4.0 * pi * r**2 * rho
    factor = mass + 4.0 * pi * r**3 * press
    denom = r * (r - 2.0 * mass)
    dpress_dr = - (rho + press) * factor / denom
    dphi_dr = factor / denom
    return (dpress_dr, dmass_dr, dphi_dr)
def tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax):
    rmin = 1e-6
    r_arr = np.linspace(rmin, rmax, npoints)
    press_c = eos_press_from_rho(rhoc, eos_Gamma, eos_kappa)
    if press_c <= 0.0:
        raise RuntimeError("Central pressure is not positive; adjust central density or EoS parameters.")
    y0 = [press_c, 0.0, 0.0]
    def rhs(r, y):
        return tov_RHS(y, r, eos_Gamma, eos_kappa)
    def event_press_zero(r, y):
        return y[0]
    event_press_zero.terminal = True
    event_press_zero.direction = -1
    sol = solve_ivp(
        rhs,
        (rmin, rmax),
        y0,
        t_eval=r_arr,
        events=event_press_zero,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    press_arr = sol.y[0]
    mass_arr = sol.y[1]
    phi_arr = sol.y[2]
    r_arr = sol.t
    idx_surf = np.argmax(press_arr <= 0)
    if idx_surf == 0:
        raise RuntimeError("Failed to integrate TOV: pressure non-positive at first grid point.")
    if idx_surf < len(r_arr):
        i1, i2 = idx_surf - 1, idx_surf
        P1, P2 = press_arr[i1], press_arr[i2]
        r1, r2 = r_arr[i1], r_arr[i2]
        m1, m2 = mass_arr[i1], mass_arr[i2]
        phi1, phi2 = phi_arr[i1], phi_arr[i2]
        frac = -P1 / (P2 - P1)
        R = r1 + frac * (r2 - r1)
        M = m1 + frac * (m2 - m1)
        phi_surface = phi1 + frac * (phi2 - phi1)
    else:
        R = r_arr[-1]
        M = mass_arr[-1]
        phi_surface = phi_arr[-1]
    if R <= 2 * M:
        raise RuntimeError("Stellar radius less than or equal to 2M: not a physical solution.")
    phi_surface_Schw = 0.5 * np.log(1.0 - 2.0 * M / R)
    delta_phi = phi_surface_Schw - phi_surface
    phi_center = phi_arr[0] + delta_phi
    lapse = np.exp(phi_center)
    star_mass = M
    return (star_mass, lapse)

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
