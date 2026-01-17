import numpy as np
from scipy import integrate, optimize

# import numpy as np
def Schroed_deriv(y, r, l, En):
    u, uprime = y
    factor = (l * (l + 1) / r**2) - (2.0 / r) + (2.0 * En)
    u2 = factor * u
    return np.array([uprime, u2])
def SolveSchroedinger(y0, En, l, R):
    """Integrate the radial Schrödinger equation inward from large r,
    then normalize the solution via Simpson's rule.
    
    Input:
      y0 : list or array of floats [u(R_max), u'(R_max)]
      En : energy (float)
      l  : angular momentum quantum number (int)
      R  : 1D numpy array of radii in ascending order
    
    Output:
      ur : normalized radial wavefunction u(r) on the grid R
    """
    R_rev = R[::-1]
    sol = integrate.solve_ivp(
        fun=lambda r, y: Schroed_deriv(y, r, l, En),
        t_span=(R_rev[0], R_rev[-1]),
        y0=y0,
        t_eval=R_rev,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )
    u_rev = sol.y[0]
    u = u_rev[::-1]
    norm = integrate.simpson(u**2, R)
    ur = u / np.sqrt(norm)
    return ur
def Shoot(En, R, l, y0):
    u = SolveSchroedinger(y0, En, l, R)
    r1, r2 = R[0], R[1]
    u1, u2 = u[0], u[1]
    f1 = u1 / (r1**l)
    f2 = u2 / (r2**l)
    slope = (f2 - f1) / (r2 - r1)
    f_at_0 = f1 - slope * r1
    return f_at_0

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
