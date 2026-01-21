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



from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
