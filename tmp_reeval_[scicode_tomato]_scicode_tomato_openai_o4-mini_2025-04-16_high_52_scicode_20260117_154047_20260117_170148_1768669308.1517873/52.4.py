import numpy as np
from scipy import integrate, optimize

# import numpy as np
def Schroed_deriv(y, r, l, En):
    u, uprime = y
    factor = (l * (l + 1) / r**2) - (2.0 / r) + (2.0 * En)
    u2 = factor * u
    return np.array([uprime, u2])
def SolveSchroedinger(y0, En, l, R):
    """Integrate the radial Schrödinger equation inward from large r,
    then normalize the solution via Simpson's rule.
    
    Input:
      y0 : list or array of floats [u(R_max), u'(R_max)]
      En : energy (float)
      l  : angular momentum quantum number (int)
      R  : 1D numpy array of radii in ascending order
    
    Output:
      ur : normalized radial wavefunction u(r) on the grid R
    """
    R_rev = R[::-1]
    sol = integrate.solve_ivp(
        fun=lambda r, y: Schroed_deriv(y, r, l, En),
        t_span=(R_rev[0], R_rev[-1]),
        y0=y0,
        t_eval=R_rev,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )
    u_rev = sol.y[0]
    u = u_rev[::-1]
    norm = integrate.simpson(u**2, R)
    ur = u / np.sqrt(norm)
    return ur
def Shoot(En, R, l, y0):
    u = SolveSchroedinger(y0, En, l, R)
    r1, r2 = R[0], R[1]
    u1, u2 = u[0], u[1]
    f1 = u1 / (r1**l)
    f2 = u2 / (r2**l)
    slope = (f2 - f1) / (r2 - r1)
    f_at_0 = f1 - slope * r1
    return f_at_0
def FindBoundStates(y0, R, l, nmax, Esearch):
    """
    Search for up to nmax bound-state energies for angular momentum l
    by finding zeros of the shooting function Shoot(E, R, l, y0)
    across the energy mesh Esearch.
    Returns a list of (l, E) tuples.
    """
    Ebnd = []
    fvals = [Shoot(E, R, l, y0) for E in Esearch]
    N = len(Esearch)
    for i in range(N - 1):
        E1, E2 = Esearch[i], Esearch[i + 1]
        f1, f2 = fvals[i], fvals[i + 1]
        if np.isnan(f1):
            continue
        if f1 == 0.0:
            Ebnd.append((l, E1))
            if len(Ebnd) >= nmax:
                break
            continue
        if not np.isnan(f2) and f1 * f2 < 0:
            try:
                E_root = optimize.brentq(lambda E: Shoot(E, R, l, y0), E1, E2)
                Ebnd.append((l, E_root))
            except ValueError:
                pass
            if len(Ebnd) >= nmax:
                break
    if len(Ebnd) < nmax and not np.isnan(fvals[-1]) and fvals[-1] == 0.0:
        Ebnd.append((l, Esearch[-1]))
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
