import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    u = y[0]
    up = y[1]
    if r == 0:
        r = 1e-8
    centrifugal = l * (l + 1) / r**2
    coulomb = -2 / r
    energy_term = -2 * En
    upp = (centrifugal + coulomb + energy_term) * u
    return [up, upp]
def SolveSchroedinger(y0, En, l, R):
    def deriv(r, y):
        return Schroed_deriv(y, r, l, En)
    R_rev = R[::-1]
    sol = integrate.solve_ivp(
        deriv, 
        (R_rev[0], R_rev[-1]),
        y0, 
        t_eval=R_rev,
        method='RK45',
        vectorized=False
    )
    u = sol.y[0][::-1]
    norm = np.sqrt(integrate.simpson(u**2, x=R))
    if norm == 0:
        ur = u
    else:
        ur = u / norm
    return ur
def Shoot(En, R, l, y0):
    u = SolveSchroedinger(y0, En, l, R)
    f1 = u[0] / (R[0] ** l)
    f2 = u[1] / (R[1] ** l)
    f_at_0 = f1 + (f2 - f1) * (-R[0]) / (R[1] - R[0])
    return f_at_0
def FindBoundStates(y0, R, l, nmax, Esearch):
    Ebnd = []
    shoots = np.array([Shoot(En, R, l, y0) for En in Esearch])
    for i in range(len(Esearch) - 1):
        f1, f2 = shoots[i], shoots[i + 1]
        if f1 * f2 < 0:
            try:
                res = optimize.root_scalar(
                    Shoot,
                    args=(R, l, y0),
                    bracket=[Esearch[i], Esearch[i+1]],
                    method='brentq',
                    xtol=1e-10
                )
                if res.converged:
                    Ebnd.append((l, res.root))
            except Exception:
                continue
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
