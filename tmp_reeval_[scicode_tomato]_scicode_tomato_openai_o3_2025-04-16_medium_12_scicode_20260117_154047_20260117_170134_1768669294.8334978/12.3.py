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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.3', 3)
target = targets[0]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,10,20), 1), target)
target = targets[1]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,10), 2), target)
target = targets[2]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,20), 3), target)
