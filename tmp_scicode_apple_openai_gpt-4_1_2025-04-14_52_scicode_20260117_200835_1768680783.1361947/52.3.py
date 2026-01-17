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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
