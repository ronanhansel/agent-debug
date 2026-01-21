import numpy as np
from scipy import integrate, optimize

def sanitize_payload(payload):
    sanitized = dict(payload)
    sanitized.pop('stop', None)
    return sanitized
def sanitize_openai_payload(payload):
    """
    Remove parameters from the payload that are not supported by the target model.
    Currently, this function strips out the `stop` parameter, which causes
    an `unsupported_parameter` error for certain chat models.

    Args:
        payload (dict): Dictionary of parameters intended for the OpenAI API.

    Returns:
        dict: A new dictionary with unsupported parameters removed.
    """
    unsupported_keys = {'stop'}
    return {k: v for k, v in payload.items() if k not in unsupported_keys}

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
