# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    m_e=9.1093837015e-31
    hbar=1.054571817e-34
    e_charge=1.602176634e-19
    eps0=8.8541878128e-12
    pi=3.141592653589793
    centrifugal=l*(l+1)/(r_grid**2)
    V_coulomb=-e_charge**2/(4.0*pi*eps0*r_grid)
    prefactor=2.0*m_e/(hbar**2)
    f_r=centrifugal+prefactor*(V_coulomb-energy)
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    """
    Given precomputed function f(r), solve the differential equation u''(r) = f(r)*u(r)
    using the Numerov method.
    Inputs:
    - f_in: 1D array of float representing f(r) at uniform grid points.
    - u_at_0: the value u(0), a float.
    - up_at_0: the derivative u'(0), a float.
    - step: the grid spacing Δr, a float.
    Output:
    - u: 1D array of float with the solution u(r) at each grid point.
    """
    N = len(f_in)
    u = [0.0 for _ in range(N)]
    u[0] = u_at_0
    if N > 1:
        u[1] = u_at_0 + step * up_at_0 + 0.5 * (step**2) * f_in[0] * u_at_0
    k = [1.0 + (step**2 / 12.0) * fi for fi in f_in]
    for n in range(1, N - 1):
        u[n+1] = (2.0 * k[n] * u[n] - k[n-1] * u[n-1]) / k[n+1]
    return u
def compute_Schrod(energy, r_grid, l):
    """
    Solve the radial Schrödinger equation u'' = f(r) u with Numerov's method
    and normalize the solution by Simpson's rule (integrating from largest r).
    
    Inputs:
      - energy: float, the energy eigenvalue E
      - r_grid: 1D numpy array of floats, the radial grid points
      - l: int, angular momentum quantum number
    Output:
      - ur_norm: 1D numpy array of floats, the normalized radial wavefunction u(r)
    """
    step = r_grid[0] - r_grid[1]
    f_r = f_Schrod(energy, l, r_grid)
    u_vals = Numerov(f_r, u_at_0=0.0, up_at_0=-1e-7, step=step)
    u = np.array(u_vals, dtype=float)
    if hasattr(integrate, 'simps'):
        simpson_func = integrate.simps
    else:
        simpson_func = integrate.simpson
    r_rev = r_grid[::-1]
    u_rev = u[::-1]
    integral = simpson_func(u_rev**2, r_rev)
    norm_const = 1.0 / np.sqrt(integral)
    ur_norm = u * norm_const
    return ur_norm
def shoot(energy, r_grid, l):
    ur_norm = compute_Schrod(energy, r_grid, l)
    r1, r2 = r_grid[-1], r_grid[-2]
    u1, u2 = ur_norm[-1], ur_norm[-2]
    f1 = u1 / (r1 ** l)
    f2 = u2 / (r2 ** l)
    f_at_0 = f1 + (-r1) * (f2 - f1) / (r2 - r1)
    return f_at_0
def find_bound_states(r_grid, l, energy_grid):
    """
    Search for up to 10 bound-state energies for angular momentum l.
    Uses shoot(energy, r_grid, l) and scipy.optimize.brentq to locate
    zeros of the shooting function across the provided energy_grid.
    
    Inputs:
      - r_grid: 1D array of radii
      - l: angular momentum quantum number (int)
      - energy_grid: 1D array of trial energies (float)
    Output:
      - bound_states: list of (l, energy_root) tuples for each bound state found
    """
    bound_states = []
    shoot_vals = [shoot(E, r_grid, l) for E in energy_grid]
    for i in range(len(energy_grid) - 1):
        f1, f2 = shoot_vals[i], shoot_vals[i + 1]
        if f1 * f2 < 0:
            try:
                root = optimize.brentq(
                    lambda E: shoot(E, r_grid, l),
                    energy_grid[i],
                    energy_grid[i + 1]
                )
                bound_states.append((l, root))
            except ValueError:
                pass
            if len(bound_states) >= 10:
                break
    return bound_states
def sort_states(bound_states):
    tie_factor = 1.0 / 10000.0
    sorted_states = sorted(bound_states, key=lambda state: state[1] + state[0] * tie_factor)
    return sorted_states
def calculate_charge_density(bound_states, r_grid, Z):
    sorted_states = sort_states(bound_states)
    charge_density = np.zeros_like(r_grid, dtype=float)
    remaining = Z
    for l, energy in sorted_states:
        if remaining <= 0:
            break
        deg = 2 * (2 * l + 1)
        occ = min(deg, remaining)
        if occ <= 0:
            continue
        u_norm = compute_Schrod(energy, r_grid, l)
        R = np.zeros_like(r_grid, dtype=float)
        nonzero = r_grid > 0
        R[nonzero] = u_norm[nonzero] / r_grid[nonzero]
        charge_density += occ * (R ** 2)
        remaining -= occ
    return charge_density
def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    """
    Solve ∇²V_H(r) = –8π ρ(r) in spherical symmetry by computing
    U(r) = r V_H(r), which satisfies U''(r) = –8π r ρ(r).
    Inputs:
      - charge_density: 1D numpy array of ρ(r) on the grid
      - u_at_0: float, U(0)
      - up_at_0: float, U'(0)
      - step: float, Δr
      - r_grid: 1D numpy array of r values
      - Z: int, atomic number (not used explicitly here)
    Output:
      - U: 1D numpy array of HartreeU(r) = r V_H(r)
    """
    N = len(r_grid)
    U = np.zeros_like(r_grid, dtype=float)
    S = -8.0 * np.pi * r_grid * charge_density
    U[0] = u_at_0
    if N > 1:
        U[1] = U[0] + step * up_at_0 + 0.5 * step**2 * S[0]
    for n in range(1, N - 1):
        U[n+1] = 2.0 * U[n] - U[n-1] + step**2 * S[n]
    return U
def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy: float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; int
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r)=V_H(r)r; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    m_e = 9.1093837015e-31
    hbar = 1.054571817e-34
    e_charge = 1.602176634e-19
    eps0 = 8.8541878128e-12
    pi = 3.141592653589793
    prefactor = 2.0 * m_e / (hbar**2)
    f_r = []
    for r, U in zip(r_grid, hartreeU):
        centrifugal = l * (l + 1) / (r**2)
        V_coulomb = -Z * e_charge**2 / (4.0 * pi * eps0 * r)
        V_H = U / r if r > 0 else 0.0
        f_r.append(centrifugal + prefactor * (V_coulomb + V_H - energy))
    return f_r
def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    """
    Solve the radial Schrödinger equation u'' = f(r) u with Numerov's method
    including the Hartree potential term, and normalize the result via Simpson's rule.
    
    Inputs:
      - energy: float, eigenvalue E (in SI units)
      - r_grid: 1D numpy array of floats, radial grid (must be descending)
      - l: int, angular momentum quantum number
      - Z: int, atomic number
      - hartreeU: 1D numpy array of floats, U(r)=r·V_H(r) (Hartree potential times r)
    Output:
      - ur_norm: 1D numpy array of floats, the normalized radial wavefunction u(r)
    """
    step = r_grid[0] - r_grid[1]
    f_r = f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    u_vals = Numerov(f_r, u_at_0=0.0, up_at_0=-1e-7, step=step)
    u = np.array(u_vals, dtype=float)
    integral = integrate.simpson(u[::-1]**2, r_grid[::-1])
    ur_norm = u / np.sqrt(integral)
    return ur_norm
def extrapolate_polyfit(energy, r_grid, l, Z, hartreeU):
    """
    Extrapolate the radial wavefunction u(r) at r = 0 by fitting a 3rd-order
    polynomial to u(r)/r^l on the first four (smallest) grid points.
    
    Inputs:
      - energy: float, the eigenvalue E
      - r_grid: 1D numpy array of radii in descending order
      - l: int, angular momentum quantum number
      - Z: int, atomic number
      - hartreeU: 1D numpy array of U(r)=r·V_H(r) (Hartree term)
    Output:
      - u0: float, extrapolated value u(r=0)
    """
    u = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    r_small = r_grid[-4:]
    u_small = u[-4:]
    f_small = u_small / (r_small ** l)
    coeffs = np.polyfit(r_small, f_small, 3)
    f0 = coeffs[-1]
    u0 = f0 if l == 0 else 0.0
    return u0
def find_bound_states_Hartree(r_grid, l, energy_grid, Z, hartreeU):
    bound_states = []
    shoot_vals = [extrapolate_polyfit(E, r_grid, l, Z, hartreeU) for E in energy_grid]
    for i in range(len(energy_grid) - 1):
        f1, f2 = shoot_vals[i], shoot_vals[i + 1]
        if f1 * f2 < 0:
            try:
                root = optimize.brentq(
                    lambda E: extrapolate_polyfit(E, r_grid, l, Z, hartreeU),
                    energy_grid[i],
                    energy_grid[i + 1]
                )
                bound_states.append((l, root))
            except ValueError:
                pass
            if len(bound_states) >= 10:
                break
    return bound_states
def calculate_charge_density_Hartree(bound_states, r_grid, Z, hartreeU):
    """
    Calculate the electron charge density and total energy (in Rydberg) 
    for a list of Hartree‐corrected bound states.

    Inputs:
      - bound_states: list of (l, energy) tuples from find_bound_states_Hartree
      - r_grid: 1D numpy array of radii (ascending or descending)
      - Z: int, atomic number (total electrons to place)
      - hartreeU: 1D numpy array U(r)=r·V_H(r) of the Hartree potential term

    Outputs:
      - charge_density: 1D numpy array of ρ(r)
      - total_energy: float, sum of state energies in Rydberg
    """
    E_RY = 2.1798723611035e-18
    sorted_states = sort_states(bound_states)
    charge_density = np.zeros_like(r_grid, dtype=float)
    total_energy = 0.0
    remaining = Z
    for l, energy in sorted_states:
        if remaining <= 0:
            break
        deg = 2 * (2 * l + 1)
        occ = min(deg, remaining)
        if occ <= 0:
            continue
        fermi_factor = occ / deg
        u_norm = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
        R = np.zeros_like(r_grid, dtype=float)
        mask = r_grid > 0
        R[mask] = u_norm[mask] / r_grid[mask]
        charge_density += occ * (R ** 2)
        E_ryd = energy / E_RY
        total_energy += E_ryd * deg * fermi_factor
        remaining -= occ
    return charge_density, total_energy
def scf_routine(r_grid, energy_grid, nmax, Z, hartreeU, tolerance, iteration):
    """
    Self-consistent field routine for the radial Schrödinger + Hartree problem.

    Inputs:
      - r_grid: 1D numpy array of radii
      - energy_grid: 1D numpy array of trial energies
      - nmax: maximum principal quantum number (we scan ℓ = 0 … nmax-1)
      - Z: atomic number (total electrons)
      - hartreeU: initial Hartree potential U(r)=r·V_H(r)
      - tolerance: convergence threshold on density
      - iteration: max number of SCF iterations

    Returns:
      - charge_density: final converged electron density (1D array)
      - total_energy: final total energy in Rydberg (float)
    """
    step = abs(r_grid[1] - r_grid[0])
    rho_old = np.zeros_like(r_grid, dtype=float)
    U_old = hartreeU.copy()
    mix = 0.5
    pi = np.pi

    for _ in range(iteration):
        bound_states = []
        for l in range(nmax):
            states_l = find_bound_states_Hartree(r_grid, l, energy_grid, Z, U_old)
            bound_states.extend(states_l)

        rho_new, E_states = calculate_charge_density_Hartree(bound_states, r_grid, Z, U_old)
        rho_mix = mix * rho_old + (1.0 - mix) * rho_new
        U_new = calculate_HartreeU(rho_mix, u_at_0=0.0, up_at_0=0.0, step=step, r_grid=r_grid, Z=Z)
        delta = np.max(np.abs(rho_mix - rho_old))
        rho_old = rho_mix
        U_old = U_new
        if delta < tolerance:
            break

    V_H = np.zeros_like(r_grid, dtype=float)
    mask = r_grid > 0
    V_H[mask] = U_old[mask] / r_grid[mask]
    integrand = rho_old * V_H * 4.0 * pi * (r_grid**2)
    E_H = 0.5 * integrate.simps(integrand, r_grid)
    E_RY = 2.1798723611035e-18
    total_energy = E_states + (E_H / E_RY)
    return rho_old, total_energy

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.14', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
r_grid = np.linspace(1e-8,20,2**14+1)
Z = 8
E0=-1.2*Z**2
energy_shift=0.5                                                                                                                        
energy_grid = -np.logspace(-4,np.log10(-E0+energy_shift),200)[::-1] + energy_shift
nmax = 5
hartreeU = -2*np.ones(len(r_grid)) + 2 * Z
tolerance = 1e-7
iteration = 10
assert cmp_tuple_or_list(scf_routine(r_grid, energy_grid, nmax, Z, hartreeU, tolerance, iteration), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
r_grid = np.linspace(1e-8,20,2**14+1)
Z = 16
E0=-1.2*Z**2
energy_shift=0.5                                                                                                                        
energy_grid = -np.logspace(-4,np.log10(-E0+energy_shift),200)[::-1] + energy_shift
nmax = 5
hartreeU = -2*np.ones(len(r_grid)) + 2 * Z
tolerance = 1e-7
iteration = 10
assert cmp_tuple_or_list(scf_routine(r_grid, energy_grid, nmax, Z, hartreeU, tolerance, iteration), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
r_grid = np.linspace(1e-8,20,2**14+1)
Z = 6
E0=-1.2*Z**2
energy_shift=0.5                                                                                                                        
energy_grid = -np.logspace(-4,np.log10(-E0+energy_shift),200)[::-1] + energy_shift
nmax = 5
hartreeU = -2*np.ones(len(r_grid)) + 2 * Z
tolerance = 1e-7
iteration = 10
assert cmp_tuple_or_list(scf_routine(r_grid, energy_grid, nmax, Z, hartreeU, tolerance, iteration), target)
