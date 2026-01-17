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
