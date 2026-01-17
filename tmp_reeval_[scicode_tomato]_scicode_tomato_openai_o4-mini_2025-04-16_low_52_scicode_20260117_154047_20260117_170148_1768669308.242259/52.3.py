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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
