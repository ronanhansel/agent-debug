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
def FindBoundStates(y0, R, l, nmax, Esearch):
    """
    Locate bound-state energies for a given angular-momentum quantum number
    using the shooting method.

    Parameters
    ----------
    y0 : sequence of two floats
        Initial values [u(R_max), u'(R_max)] supplied to SolveSchroedinger.
    R : 1-D array_like
        Radial grid (all positive). Integration starts at R[0] (largest r).
    l : int
        Orbital angular-momentum quantum number.
    nmax : int
        Maximum number of bound states to return.
    Esearch : 1-D array_like
        Energy mesh (typically negative). Consecutive points define search
        intervals; a change of sign of the shooting function between two points
        brackets a bound-state energy.

    Returns
    -------
    Ebnd : list of tuple
        Each element is (l, E_bound). The list is sorted by increasing energy
        (most negative first) and contains at most `nmax` states.
    """
    if nmax <= 0:
        return []
    R = np.asarray(R, dtype=float)
    if R.ndim != 1 or np.any(R <= 0):
        raise ValueError("R must be a 1-D array of positive radii.")
    Egrid = np.asarray(Esearch, dtype=float)
    if Egrid.ndim != 1 or Egrid.size < 2:
        raise ValueError("Esearch must be a 1-D array with at least two points.")
    def _shoot_safe(E):
        try:
            return Shoot(E, R, l, y0)
        except Exception:
            return np.nan
    Fvals = np.array([_shoot_safe(E) for E in Egrid])
    Ebnd = []
    zero_tol = 1e-8
    dup_tol = 1e-10
    for i in range(len(Egrid) - 1):
        if len(Ebnd) >= nmax:
            break
        E1, E2 = Egrid[i], Egrid[i + 1]
        F1, F2 = Fvals[i], Fvals[i + 1]
        if np.isnan(F1) or np.isnan(F2):
            continue
        root_found = False
        if abs(F1) < zero_tol:
            root = E1
            root_found = True
        elif abs(F2) < zero_tol:
            root = E2
            root_found = True
        elif F1 * F2 < 0.0:
            Elo, Ehi = (E1, E2) if E1 < E2 else (E2, E1)
            try:
                root = optimize.brentq(lambda E: Shoot(E, R, l, y0),
                                       Elo, Ehi, xtol=1e-12, rtol=1e-10,
                                       maxiter=200)
                root_found = True
            except (ValueError, RuntimeError):
                continue
        if root_found:
            if all(abs(root - prev_E) > dup_tol for _, prev_E in Ebnd):
                Ebnd.append((int(l), float(root)))
                if len(Ebnd) >= nmax:
                    break
    Ebnd.sort(key=lambda t: t[1])
    return Ebnd

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.4', 3)
target = targets[0]

y0 = [0, -1e-5]
Esearch = -1.2/np.arange(1,20,0.2)**2
R = np.logspace(-6,2.2,500)
nmax=7
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
assert np.allclose(Bnd, target)
target = targets[1]

Esearch = -0.9/np.arange(1,20,0.2)**2
R = np.logspace(-8,2.2,1000)
nmax=5
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
assert np.allclose(Bnd, target)
target = targets[2]

Esearch = -1.2/np.arange(1,20,0.2)**2
R = np.logspace(-6,2.2,500)
nmax=7
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0,R,l,nmax-l,Esearch)
assert np.isclose(Bnd[0], (0,-1)).any() == target.any()
