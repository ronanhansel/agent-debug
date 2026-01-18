import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    u, uprime = y
    Z = 1.0
    coeff = l * (l + 1) / r**2 - 2 * Z / r - 2 * En
    u2 = coeff * u
    return np.array([uprime, u2], dtype=float)
def SolveSchroedinger(y0, En, l, R):
    R_rev = R[::-1]
    sol = integrate.solve_ivp(fun=lambda r, y: Schroed_deriv(y, r, l, En), t_span=(R_rev[0], R_rev[-1]), y0=y0, t_eval=R_rev, vectorized=False)
    u = sol.y[0][::-1]
    norm = integrate.simpson(u**2, x=R)
    ur = u / np.sqrt(norm)
    return ur
def Shoot(En, R, l, y0):
    """Extrapolate u(0) based on results from SolveSchroedinger function
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    Output 
    f_at_0: Extrapolate u(0), float
    """
    ur = SolveSchroedinger(y0, En, l, R)
    f = ur / (R**l)
    r0, r1 = R[0], R[1]
    f0, f1 = f[0], f[1]
    slope = (f1 - f0) / (r1 - r0)
    f_at_0 = f0 - slope * r0
    return f_at_0
def FindBoundStates(y0, R, l, nmax, Esearch):
    Ebnd = []
    f_values = [Shoot(E, R, l, y0) for E in Esearch]
    for i in range(len(Esearch) - 1):
        f1, f2 = f_values[i], f_values[i + 1]
        if f1 == 0.0:
            Ebnd.append((l, Esearch[i]))
        elif f1 * f2 < 0.0:
            sol = optimize.root_scalar(
                lambda E: Shoot(E, R, l, y0),
                bracket=(Esearch[i], Esearch[i + 1]),
                method='brentq'
            )
            if sol.converged:
                Ebnd.append((l, sol.root))
        if len(Ebnd) >= nmax:
            break
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
