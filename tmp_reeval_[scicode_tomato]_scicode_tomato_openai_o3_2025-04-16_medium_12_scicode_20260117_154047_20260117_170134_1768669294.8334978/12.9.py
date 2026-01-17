# Compatibility shim for scipy.integrate.simps (deprecated in SciPy 1.14, removed in 1.17)
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    f_r = []
    eps = 1e-15
    for r in r_grid:
        r_safe = r if r != 0.0 else eps
        f_val = (l * (l + 1)) / (r_safe ** 2) - 2.0 / r_safe - 2.0 * energy
        f_r.append(f_val)
    return f_r
def Numerov(f_in, u_at_0, up_at_0, step):
    '''
    Solve u''(r) = f(r)·u(r) on an equally-spaced radial grid by the Numerov method.
    Parameters
    ----------
    f_in : (N,) array_like
        Values of f(r) on a uniform grid r = n*step.
    u_at_0 : float
        Boundary condition u(r=0).
    up_at_0 : float
        Boundary condition u'(r=0).
    step : float
        Grid spacing h.
    Returns
    -------
    u : list, length N
        Numerical solution u(r) on the same grid.
    '''
    f_in = [float(v) for v in f_in]
    N = len(f_in)
    h = float(step)
    u = [0.0] * N
    u[0] = u_at_0
    if N == 1:
        return u
    h2 = h * h
    u[1] = u_at_0 + h * up_at_0 + 0.5 * h2 * f_in[0] * u_at_0
    if N == 2:
        return u
    h2_over_12 = h2 / 12.0
    for i in range(1, N - 1):
        k_im1 = f_in[i - 1]
        k_i = f_in[i]
        k_ip1 = f_in[i + 1]
        numerator = (2.0 * u[i] * (1.0 - 5.0 * h2_over_12 * k_i)
                     - u[i - 1] * (1.0 + h2_over_12 * k_im1))
        denominator = 1.0 + h2_over_12 * k_ip1
        u[i + 1] = numerator / denominator
    return u
def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array-like of float
    l: angular momentum quantum number; an int
    
    Output
    ur_norm: 1-D NumPy array – normalized radial wave-function u(r)
    '''
    # --- prepare radial grid -------------------------------------------------
    r_grid = np.asarray(r_grid, dtype=float)
    if r_grid.size < 2:
        raise ValueError("r_grid must contain at least two points.")

    # Determine whether the supplied grid is ascending or descending
    ascending = r_grid[1] > r_grid[0]

    # We have to start the integration at the largest radius.
    # If the grid is ascending, reverse it so r_work[0] is the largest r.
    if ascending:
        r_work = r_grid[::-1]
    else:
        r_work = r_grid

    # Step for Numerov: difference of the first two (now consecutive) points
    step = r_work[0] - r_work[1]
    if step == 0.0:
        raise ValueError("Consecutive r_grid points are identical; step size is zero.")
    # The step must be positive for the Numerov routine (it only uses h² except
    # in the first-step formula, where the sign would matter).  Because we have
    # arranged r_work[0] > r_work[1], this difference is positive.
    if step < 0:
        step = abs(step)

    # --- build f(r) and integrate -------------------------------------------
    f_r = f_Schrod(energy, l, r_work)                # f(r) on working grid
    u_work = Numerov(f_r, u_at_0=0.0, up_at_0=-1e-7, step=step)
    u_work = np.asarray(u_work, dtype=float)

    # Restore original ordering if the grid was reversed
    if ascending:
        u = u_work[::-1]
    else:
        u = u_work

    # --- normalisation -------------------------------------------------------
    # Simpson integration; abs to cope with possible descending x-array
    norm_sq = integrate.simps(u**2, r_grid)
    norm = np.sqrt(abs(norm_sq))
    if norm == 0.0:
        raise ValueError("Wave-function has zero norm and cannot be normalised.")

    ur_norm = u / norm
    return ur_norm
def shoot(energy, r_grid, l):
    """Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    
    Output 
    f_at_0: float
        Linear–extrapolated value, at r = 0, of
        f(r) = u(r) / r**l   where u(r) is the (normalised) radial
        wave-function obtained from `compute_Schrod`.
    """
    r_grid = [float(r) for r in r_grid]
    if len(r_grid) < 2:
        raise ValueError("r_grid must contain at least two points.")
    u_vals = compute_Schrod(energy, r_grid, l)
    r1, r2 = float(r_grid[0]), float(r_grid[1])
    u1, u2 = float(u_vals[0]), float(u_vals[1])
    if r1 == 0.0 or r2 == 0.0:
        raise ValueError("The first two radial grid points must be non-zero.")
    if r1 == r2:
        raise ValueError("The first two radial grid points must be distinct.")
    f1 = u1 / (r1 ** l) if l else u1
    f2 = u2 / (r2 ** l) if l else u2
    slope = (f2 - f1) / (r2 - r1)
    f_at_0 = f1 - slope * r1
    return float(f_at_0)
def find_bound_states(r_grid, l, energy_grid):
    '''
    Search (via shooting and root-bracketing) for bound–state energies of a
    hydrogen-like atom with angular-momentum quantum number ``l``.

    Parameters
    ----------
    r_grid : 1-D array_like of float
        Radial mesh (need not be sorted).
    l : int
        Angular-momentum quantum number.
    energy_grid : 1-D array_like of float
        Trial energies (Hartree).  Consecutive points will be inspected for a
        sign-change in the shooting function so they should span the expected
        bound-state region (typically negative energies).

    Returns
    -------
    bound_states : list of tuple
        Up to ten tuples ``(l, E)`` containing the quantum number ``l`` and the
        eigen-energy ``E`` (Hartree) of each bound state found, sorted in
        ascending energy (more negative first).
    '''
    r_grid = np.asarray(r_grid, dtype=float)
    if r_grid.size < 2:
        raise ValueError('r_grid must contain at least two points.')
    energies = np.asarray(energy_grid, dtype=float)
    if energies.size < 2:
        raise ValueError('energy_grid must contain at least two points.')
    energies = np.sort(energies)
    shoot_vals = []
    for E in energies:
        try:
            shoot_vals.append(shoot(float(E), r_grid, int(l)))
        except Exception:
            shoot_vals.append(np.nan)
    bound_states = []
    MAX_FOUND = 10
    DUPL_TOL   = 1.0e-6
    for i in range(len(energies) - 1):
        f1, f2 = shoot_vals[i], shoot_vals[i + 1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        E1, E2 = float(energies[i]), float(energies[i + 1])
        root_E = None
        if abs(f1) < 1e-12:
            root_E = E1
        elif f1 * f2 < 0.0:
            try:
                root_E = optimize.brentq(
                    lambda E, rg=r_grid, lq=l: shoot(E, rg, lq),
                    E1, E2, xtol=1e-12, rtol=1e-10, maxiter=100
                )
            except (ValueError, RuntimeError):
                continue
        if root_E is not None:
            if all(abs(root_E - prev_E) > DUPL_TOL for _, prev_E in bound_states):
                bound_states.append((int(l), float(root_E)))
                if len(bound_states) >= MAX_FOUND:
                    break
    bound_states.sort(key=lambda t: t[1])
    return bound_states
def sort_states(bound_states):
    '''
    Input
    -----
    bound_states : list[tuple[int, float]]
        Each element is a tuple (l, E) with angular-momentum quantum number
        `l` and energy `E` (Hartree).

    Output
    ------
    sorted_states : list[tuple[int, float]]
        The same states sorted so that
          1) lower energy (more negative) appears first;
          2) if two energies are identical – or so close that their difference
             is much smaller than 1 × 10⁻⁴ Ha – the state with the smaller
             angular-momentum quantum number comes first.  This is enforced by
             adding the quantity l/10000 to the energy when building the sort
             key, thereby letting `l` influence the order only at the
             1 × 10⁻⁴ level.
    '''
    if not bound_states:
        return []
    scale = 1.0 / 10000.0
    return sorted(bound_states, key=lambda st: st[1] + st[0] * scale)
def calculate_charge_density(bound_states, r_grid, Z):
    r_grid = np.asarray(r_grid, dtype=float)
    if r_grid.size == 0:
        raise ValueError("r_grid must contain at least one point.")
    if Z < 0:
        raise ValueError("Atomic number Z must be non-negative.")
    states_sorted = sort_states(bound_states)
    charge_density = np.zeros_like(r_grid, dtype=float)
    eps = 1.0e-30
    denom = 4.0 * np.pi * np.where(r_grid == 0.0, eps, r_grid**2)
    electrons_remaining = int(Z)
    for l, energy in states_sorted:
        if electrons_remaining <= 0:
            break
        l_int = int(l)
        degeneracy = 2 * (2 * l_int + 1)
        occupy = min(degeneracy, electrons_remaining)
        if occupy == 0:
            continue
        try:
            u_r = compute_Schrod(float(energy), r_grid, l_int)
        except Exception:
            continue
        charge_density += occupy * (u_r**2) / denom
        electrons_remaining -= occupy
    return charge_density
def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    #
    # Replace NumPy operations with pure Python equivalents
    #
    r_grid = [float(v) for v in r_grid]  # ensure list of floats
    rho    = [float(v) for v in charge_density]

    if len(r_grid) != len(rho):
        raise ValueError("charge_density and r_grid must have identical length.")
    if len(r_grid) == 0:
        raise ValueError("r_grid must not be empty.")
    if step == 0.0:
        raise ValueError("step must be non-zero.")

    h  = abs(float(step))
    h2 = h * h

    ascending = (len(r_grid) == 1) or (r_grid[1] > r_grid[0])
    if ascending:
        r_work  = r_grid
        rho_work = rho
    else:
        r_work  = list(reversed(r_grid))
        rho_work = list(reversed(rho))

    N = len(r_work)
    pi = 3.141592653589793

    s = [-8.0 * pi * r_work[i] * rho_work[i] for i in range(N)]

    U = [0.0] * N
    U[0] = float(u_at_0)

    if N > 1:
        U[1] = U[0] + h * float(up_at_0) + 0.5 * h2 * s[0]
        coeff = h2 / 12.0
        for i in range(1, N - 1):
            U[i + 1] = (2.0 * U[i] - U[i - 1]
                        + coeff * (s[i + 1] + 10.0 * s[i] + s[i - 1]))

    x = U if ascending else list(reversed(U))
    return x
def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    """
    Return f(r) for the radial Schrödinger equation including the Hartree term.

    In atomic units the radial equation for u(r)=r·R(r) reads
        u''(r) = f(r) · u(r)
    with
        f(r) = l(l+1)/r² − 2·Z/r + 2·V_H(r) − 2·energy ,
    where the Hartree potential V_H(r) is supplied through U(r)=V_H(r)·r.

    Parameters
    ----------
    energy   : float
        Eigen-energy (Hartree) to be tested.
    r_grid   : 1-D sequence of float
        Radial grid points.
    l        : int
        Orbital angular-momentum quantum number.
    Z        : int
        Nuclear charge.
    hartreeU : 1-D sequence of float
        Values of U(r)=V_H(r)·r on the same radial grid.

    Returns
    -------
    f_r : list[float]
        Values of f(r) corresponding to each point in `r_grid`.
    """
    if len(r_grid) != len(hartreeU):  # Ensure arrays are compatible
        raise ValueError("r_grid and hartreeU must have identical length.")
    eps = 1.0e-15
    l_term = l * (l + 1)
    f_r = []
    for r, U in zip(r_grid, hartreeU):
        r_val = float(r)
        u_val = float(U)
        r_safe = r_val if r_val != 0.0 else eps
        V_H = u_val / r_safe
        f_val = (
            l_term / (r_safe ** 2)
            - 2.0 * Z / r_safe
            + 2.0 * V_H
            - 2.0 * energy
        )
        f_r.append(float(f_val))
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
