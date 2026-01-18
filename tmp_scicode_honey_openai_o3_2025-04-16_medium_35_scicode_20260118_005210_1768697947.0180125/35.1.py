import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    def is_iterable(obj):
        try:
            iter(obj)
            return not isinstance(obj, (str, bytes))
        except TypeError:
            return False

    def compute_lambda(l_val, m_rel):
        m_e = 9.109e-31
        h = 6.626e-34
        c = 3.0e8
        L_m = l_val * 1e-9
        m_eff = m_rel * m_e
        E1 = h ** 2 / (8.0 * m_eff * L_m ** 2)
        return (h * c) / E1 * 1e9

    if not is_iterable(L) and not is_iterable(mr):
        return compute_lambda(float(L), float(mr))

    if is_iterable(L) and not is_iterable(mr):
        mr_iter = [mr] * len(L)
    elif not is_iterable(L) and is_iterable(mr):
        L = [L] * len(mr)
        mr_iter = mr
    else:
        mr_iter = mr

    return [compute_lambda(float(l), float(m_rel)) for l, m_rel in zip(L, mr_iter)]

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.1', 3)
target = targets[0]

assert np.allclose(ground_state_wavelength(5,0.6), target)
target = targets[1]

assert np.allclose(ground_state_wavelength(10,0.6), target)
target = targets[2]

assert np.allclose(ground_state_wavelength(10,0.06), target)
