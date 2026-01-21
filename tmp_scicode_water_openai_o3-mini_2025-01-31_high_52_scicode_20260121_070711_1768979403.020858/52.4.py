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
def remove_unsupported_stop_parameter(request_params):
    if isinstance(request_params, dict) and 'stop' in request_params:
        request_params = request_params.copy()
        request_params.pop('stop')
    return request_params

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.4', 3)
target = targets[0]

y0 = [0, -1e-5]
Esearch = -1.2/np.arange(1,20,0.2)**2
R = np.logspace(-6,2.2,500)
nmax=7
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
assert np.allclose(Bnd, target)
target = targets[1]

Esearch = -0.9/np.arange(1,20,0.2)**2
R = np.logspace(-8,2.2,1000)
nmax=5
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
assert np.allclose(Bnd, target)
target = targets[2]

Esearch = -1.2/np.arange(1,20,0.2)**2
R = np.logspace(-6,2.2,500)
nmax=7
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(y0,R,l,nmax-l,Esearch)
assert np.isclose(Bnd[0], (0,-1)).any() == target.any()
