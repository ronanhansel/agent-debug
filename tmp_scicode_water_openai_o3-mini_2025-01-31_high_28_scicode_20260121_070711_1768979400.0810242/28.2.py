import numpy as np
from scipy.integrate import simpson

def remove_unsupported_parameters(request_params):
    return {k: v for k, v in request_params.items() if k != 'stop'}
def remove_unsupported_stop_parameter(request_params):
    """
    Returns a copy of `request_params` with the unsupported 'stop' parameter removed.
    
    Args:
        request_params (dict): A dictionary of parameters intended for an API request.
    
    Returns:
        dict: A new dictionary identical to `request_params` but without the 'stop' key.
    """
    if not isinstance(request_params, dict):
        raise TypeError("request_params must be a dictionary")
    filtered_params = dict(request_params)
    filtered_params.pop('stop', None)
    return filtered_params

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('28.2', 3)
target = targets[0]

lambda_ = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 200  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 150 # initial waist position
# User input for beam quality index
Mp2 = 1.5
assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)
target = targets[1]

lambda_ = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 180 # initial waist position
# User input for beam quality index
Mp2 = 1.5
assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)
target = targets[2]

lambda_ = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 100 # initial waist position
# User input for beam quality index
Mp2 = 1.5
assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)
