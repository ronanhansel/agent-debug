import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    u, up = y
    upp = (l*(l+1)/r**2 - 2.0/r - 2.0*En) * u
    return np.array([up, upp])
def SolveSchroedinger(y0, En, l, R):
    '''Integrate the derivative of y within a certain radius
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  a 1d array (linspace) of radius (float)
    Output
    ur: the normalized radial solution u(r), 1d numpy array
    '''
    R_rev = R[::-1]
    sol = integrate.odeint(Schroed_deriv, y0, R_rev, args=(l, En))
    u = sol[:, 0][::-1]
    norm = np.sqrt(integrate.simpson(u**2, R))
    return u / norm
def Shoot(En, R, l, y0):
    '''Extrapolate u(0) based on results from SolveSchroedinger function
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    Output 
    f_at_0: Extrapolated u(0), float
    '''
    u = SolveSchroedinger(y0, En, l, R)
    r1, r2 = R[0], R[1]
    u1, u2 = u[0], u[1]
    f1 = u1 / (r1**l)
    f2 = u2 / (r2**l)
    f_at_0 = f1 - (r1 / (r2 - r1)) * (f2 - f1)
    return f_at_0
def FindBoundStates(y0, R, l, nmax, Esearch):
    '''Input
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    nmax: maximum number of bound states wanted, int
    Esearch: energy mesh used for search, an 1D array of float
    Output
    Ebnd: a list of tuples (l, energy) of all bound states found, up to nmax
    '''
    Ebnd = []
    tol = 1e-8
    fvals = [Shoot(E, R, l, y0) for E in Esearch]
    for idx, f in enumerate(fvals):
        if abs(f) < tol:
            E0 = Esearch[idx]
            if not any(abs(E0 - root[1]) < tol for root in Ebnd):
                Ebnd.append((l, E0))
                if len(Ebnd) >= nmax:
                    return Ebnd
    for i in range(len(Esearch) - 1):
        f1, f2 = fvals[i], fvals[i + 1]
        if f1 * f2 < 0.0:
            E1, E2 = Esearch[i], Esearch[i + 1]
            try:
                Eroot = optimize.brentq(lambda E: Shoot(E, R, l, y0), E1, E2)
            except ValueError:
                continue
            if not any(abs(Eroot - root[1]) < tol for root in Ebnd):
                Ebnd.append((l, Eroot))
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
