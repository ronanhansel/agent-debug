# Compatibility shims for deprecated NumPy/SciPy APIs
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
# Note: scipy.integrate.simps -> simpson shim handled at import level
try:
    from scipy import integrate as _sci_int
    if not hasattr(_sci_int, 'simps'):
        _sci_int.simps = _sci_int.simpson
except ImportError:
    pass

from scipy import integrate
from scipy import optimize
import numpy as np

def required_function(*args, **kwargs):
    raise NotImplementedError("This function should be implemented in child classes.")
def required_function(*args, **kwargs):
    #
    pass

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.2', 3)
target = targets[0]

assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[1]

assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[2]

assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
