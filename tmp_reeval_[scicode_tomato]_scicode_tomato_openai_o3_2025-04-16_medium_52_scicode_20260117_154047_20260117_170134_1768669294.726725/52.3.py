import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    '''
    Calculate the derivative dy/dr for the radial Schrödinger equation of a
    hydrogen-like atom (nuclear charge Z = 1) rewritten as a first-order system.

    Parameters
    ----------
    y : array_like of length 2
        y = [u, u'] where
            u   : radial wave-function value at radius r,
            u'  : first derivative du/dr at radius r.
    r : float
        Radial coordinate.
    l : int
        Orbital angular-momentum quantum number.
    En : float
        Energy eigenvalue (in atomic units).

    Returns
    -------
    numpy.ndarray of length 2
        dy/dr = [u', u''] at the given radius.
    '''
    u, up = y  # unpack current values
    if r == 0.0:
        coeff = 0.0
    else:
        coeff = l * (l + 1) / r**2 - 2.0 / r + 2.0 * En
    u_dd = coeff * u  # second derivative from radial Schrödinger equation
    return np.array([up, u_dd])  # np is assumed to be available in the execution environment
def SolveSchroedinger(y0, En, l, R):
    R = np.asarray(R, dtype=float)
    if R[0] < R[-1]:
        R_int = R[::-1]
        flip_back = True
    else:
        R_int = R
        flip_back = False
    sol = integrate.solve_ivp(
        fun=lambda r, y: Schroed_deriv(y, r, l, En),
        t_span=(R_int[0], R_int[-1]),
        y0=np.asarray(y0, dtype=float),
        t_eval=R_int,
        rtol=1e-10,
        atol=1e-10
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    u = sol.y[0]
    if flip_back:
        u = u[::-1]
    norm = integrate.simpson(u**2, x=R)
    if norm <= 0.0:
        raise ValueError("Non-positive norm encountered during normalisation.")
    u /= np.sqrt(norm)
    return u
def Shoot(En, R, l, y0):
    '''Extrapolate the regularised radial wave-function to r = 0.

    Parameters
    ----------
    En : float
        Trial energy eigenvalue.
    R : array_like
        Radial grid (must contain at least two positive radii; order arbitrary).
    l : int
        Orbital angular-momentum quantum number.
    y0 : array_like, shape (2,)
        Initial conditions [u(R_max), u'(R_max)] for the ODE integration.

    Returns
    -------
    float
        Linearly-extrapolated value of  f(r) = u(r) / r**l  at r = 0.
    '''
    R = np.asarray(R, dtype=float)
    if R.size < 2:
        raise ValueError("Radius grid R must contain at least two points.")
    if np.any(R <= 0):
        raise ValueError("All radii must be positive to evaluate r**l.")
    u = SolveSchroedinger(y0=y0, En=En, l=l, R=R)
    f = u / (R ** l)
    idx = np.argsort(R)[:2]
    r1, r2 = R[idx]
    f1, f2 = f[idx]
    if np.isclose(r1, r2):
        raise ValueError("The first two radial points must be distinct for extrapolation.")
    f_at_0 = f1 - (f2 - f1) * r1 / (r2 - r1)
    return float(f_at_0)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
