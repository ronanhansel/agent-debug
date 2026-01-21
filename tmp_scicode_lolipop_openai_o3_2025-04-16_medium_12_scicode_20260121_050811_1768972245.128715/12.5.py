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
    raise NotImplementedError
def _build_layers(self, *args, **kwargs):
    pass
def execute(*args, **kwargs):
    if args:
        return args[0]
    return None


from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.5', 3)
target = targets[0]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),2, -1.2/np.arange(1,20,0.2)**2), target)
target = targets[1]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),3,-1.2/np.arange(1,20,0.2)**2), target)
target = targets[2]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),0,-1.2/np.arange(1,20,0.2)**2), target)
