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

def implemented_method(*args, **kwargs):
    return "Implemented"

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.6', 3)
target = targets[0]

bound_states=[]
for l in range(6):
    bound_states += find_bound_states(np.linspace(1e-8,100,2000),l,-1.2/np.arange(1,20,0.2)**2)
assert np.allclose(sort_states(bound_states), target)
target = targets[1]

bound_states=[]
for l in range(3):
    bound_states += find_bound_states(np.linspace(1e-8,100,2000),l,-1.2/np.arange(1,20,0.2)**2)
assert np.allclose(sort_states(bound_states), target)
target = targets[2]

bound_states=[]
for l in range(1):
    bound_states += find_bound_states(np.linspace(1e-8,100,2000),l,-1.2/np.arange(1,20,0.2)**2)
assert np.allclose(sort_states(bound_states), target)
