import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    '''
    Calculate the derivative dy/dr for the radial Schrödinger equation of a
    hydrogen-like atom (nuclear charge Z = 1) rewritten as a first-order system.

    Parameters
    ----------
    y : array_like of length 2
        y = [u, u'] where
            u   : radial wave-function value at radius r,
            u'  : first derivative du/dr at radius r.
    r : float
        Radial coordinate.
    l : int
        Orbital angular-momentum quantum number.
    En : float
        Energy eigenvalue (in atomic units).

    Returns
    -------
    numpy.ndarray of length 2
        dy/dr = [u', u''] at the given radius.
    '''
    u, up = y  # unpack current values
    if r == 0.0:
        coeff = 0.0
    else:
        coeff = l * (l + 1) / r**2 - 2.0 / r + 2.0 * En
    u_dd = coeff * u  # second derivative from radial Schrödinger equation
    return np.array([up, u_dd])  # np is assumed to be available in the execution environment

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.1', 3)
target = targets[0]

y = [0.0,-1e-5]
r = 100
l = 1
En = 1
assert np.allclose(Schroed_deriv(y,r,l,En), target)
target = targets[1]

y = [0.0,-2e-5]
r = 1.1
l = 2
En = 1.5
assert np.allclose(Schroed_deriv(y,r,l,En), target)
target = targets[2]

y = [0.0,-2e-5]
r = 3
l = 1
En = 5
assert np.allclose(Schroed_deriv(y,r,l,En), target)
