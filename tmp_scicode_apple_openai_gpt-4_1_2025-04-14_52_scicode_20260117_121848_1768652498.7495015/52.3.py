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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
