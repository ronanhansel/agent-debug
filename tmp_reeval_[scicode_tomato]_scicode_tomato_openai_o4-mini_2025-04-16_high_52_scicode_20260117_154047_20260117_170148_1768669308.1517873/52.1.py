import numpy as np
from scipy import integrate, optimize

# import numpy as np
def Schroed_deriv(y, r, l, En):
    u, uprime = y
    factor = (l * (l + 1) / r**2) - (2.0 / r) + (2.0 * En)
    u2 = factor * u
    return np.array([uprime, u2])

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
