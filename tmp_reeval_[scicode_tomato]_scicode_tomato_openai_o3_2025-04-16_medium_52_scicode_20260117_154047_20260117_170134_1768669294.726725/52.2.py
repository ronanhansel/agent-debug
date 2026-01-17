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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.2', 3)
target = targets[0]

y0 = [0, -1e-5]
En = 1.0
l = 1
R = np.logspace(-2,3.2,100)
assert np.allclose(SolveSchroedinger(y0,En,l,R), target)
target = targets[1]

y0 = [0, -1e-5]
En = 1.5
l = 2
R = np.logspace(-1,3.2,100)
assert np.allclose(SolveSchroedinger(y0,En,l,R), target)
target = targets[2]

y0 = [0, -1e-5]
En = 2.5
l = 2
R = np.logspace(1,3.2,100)
assert np.allclose(SolveSchroedinger(y0,En,l,R), target)
