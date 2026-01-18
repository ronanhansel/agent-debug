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
