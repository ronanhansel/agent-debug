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
