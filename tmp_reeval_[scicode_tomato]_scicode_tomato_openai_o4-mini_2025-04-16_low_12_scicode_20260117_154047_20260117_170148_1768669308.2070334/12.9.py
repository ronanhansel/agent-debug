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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.9', 3)
target = targets[0]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=28
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(f_Schrod_Hartree(-0.5, r_grid, 2, Z, hu), target)
target = targets[1]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 3
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(f_Schrod_Hartree(-0.4, r_grid, 2, Z, hu), target)
target = targets[2]

energy_grid = -0.9/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(f_Schrod_Hartree(-0.5, r_grid, 3, Z, hu), target)
