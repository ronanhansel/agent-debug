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
def required_method(self, *args, **kwargs):
    return None
def required_method(*args, **kwargs):
    raise NotImplementedError()  # This method must be implemented in child classes

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.8', 3)
target = targets[0]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=28
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
assert np.allclose(calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z), target)
target = targets[1]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 3
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
assert np.allclose(calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z), target)
target = targets[2]

energy_grid = -0.9/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
assert np.allclose(calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z), target)
