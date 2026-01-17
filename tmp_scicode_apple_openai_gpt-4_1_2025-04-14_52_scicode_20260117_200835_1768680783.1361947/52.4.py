import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    u = y[0]
    up = y[1]
    if r == 0:
        upp = 0.0
    else:
        upp = (l * (l + 1) / r**2 - 2.0 / r + 2.0 * En) * u
    return [up, upp]
def SolveSchroedinger(y0, En, l, R):
    R_rev = R[::-1]
    def odesys(r, y):
        return Schroed_deriv(y, r, l, En)
    sol = integrate.solve_ivp(
        odesys,
        (R_rev[0], R_rev[-1]),
        y0,
        t_eval=R_rev,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    u = sol.y[0][::-1]
    norm = np.sqrt(integrate.simpson(np.abs(u)**2, R))
    ur = u / norm
    return ur
def Shoot(En, R, l, y0):
    ur = SolveSchroedinger(y0, En, l, R)
    r1, r2 = R[0], R[1]
    u1, u2 = ur[0], ur[1]
    f1 = u1 / (r1**l) if l > 0 else u1
    f2 = u2 / (r2**l) if l > 0 else u2
    f_at_0 = f1 + (f2 - f1) * (-r1) / (r2 - r1)
    return f_at_0
def FindBoundStates(y0, R, l, nmax, Esearch):
    Ebnd = []
    fvals = [Shoot(En, R, l, y0) for En in Esearch]
    fvals = np.array(fvals)
    n_found = 0
    for i in range(len(Esearch) - 1):
        f1, f2 = fvals[i], fvals[i+1]
        if f1 == 0.0:
            continue
        if f1 * f2 < 0:
            try:
                E_root = optimize.brentq(
                    lambda E: Shoot(E, R, l, y0),
                    Esearch[i], Esearch[i+1],
                    xtol=1e-10, rtol=1e-8, maxiter=100
                )
                Ebnd.append((l, E_root))
                n_found += 1
                if n_found >= nmax:
                    break
            except ValueError:
                continue
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
