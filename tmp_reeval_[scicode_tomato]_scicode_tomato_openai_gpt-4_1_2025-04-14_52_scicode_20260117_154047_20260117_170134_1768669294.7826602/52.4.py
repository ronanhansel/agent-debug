import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    u = y[0]
    up = y[1]
    if r == 0:
        r = 1e-12 
    u2p = (l * (l + 1) / r**2 + 2 / r + 2 * En) * u
    return [up, u2p]
def SolveSchroedinger(y0, En, l, R):
    Rint = R
    y0int = y0

    def sys(r, y):
        return Schroed_deriv(y, r, l, En)
    
    sol = integrate.solve_ivp(
        sys,
        (Rint[0], Rint[-1]),
        y0int,
        t_eval=Rint,
        method='RK45',
        vectorized=False
    )

    u_sol = sol.y[0]

    if np.all(np.abs(u_sol) < 1e-12):
        ur = u_sol
    else:
        norm2 = integrate.simpson(np.abs(u_sol)**2, Rint)
        if (norm2 is not None) and (norm2 > 0) and (not np.isnan(norm2)):
            ur = u_sol / np.sqrt(norm2)
        else:
            ur = u_sol

    return ur
def Shoot(En, R, l, y0):
    u = SolveSchroedinger(y0, En, l, R)
    r1 = R[0]
    r2 = R[1]
    f1 = u[0] / (r1 ** l)
    f2 = u[1] / (r2 ** l)
    a = (f2 - f1) / (r2 - r1)
    f_at_0 = f1 - a * r1
    return f_at_0
def FindBoundStates(y0, R, l, nmax, Esearch):
    shoot_vals = np.array([Shoot(E, R, l, y0) for E in Esearch])
    Ebnd = []
    sgn = np.sign(shoot_vals)
    crossings = np.where(np.diff(sgn))[0]
    for idx in crossings:
        if len(Ebnd) >= nmax:
            break
        E1, E2 = Esearch[idx], Esearch[idx + 1]
        f1, f2 = shoot_vals[idx], shoot_vals[idx + 1]
        if f1 == f2:
            continue
        try:
            res = optimize.root_scalar(lambda E: Shoot(E, R, l, y0),
                                      bracket=[E1, E2], method='brentq')
            if res.converged:
                energy = res.root
                if not any(abs(energy - e[1]) < 1e-8 for e in Ebnd):
                    Ebnd.append((l, energy))
        except Exception:
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
