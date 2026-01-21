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
def apply_stop_sequences(text, stop=None):
    if not stop:
        return text
    earliest_index = len(text)
    for sequence in stop:
        idx = text.find(sequence)
        if idx != -1 and idx < earliest_index:
            earliest_index = idx
    return text[:earliest_index]

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
