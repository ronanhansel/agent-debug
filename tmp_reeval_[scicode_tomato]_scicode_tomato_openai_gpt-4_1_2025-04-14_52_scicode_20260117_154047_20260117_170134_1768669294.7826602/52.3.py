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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
