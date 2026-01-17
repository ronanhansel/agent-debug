import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, args):
    """
    Input:
        dim: int or list of ints
            Dimension(s) of the ket space.
        args: int or list of ints
            Index (or indices) of the basis vector(s).
    Output:
        out: numpy.ndarray
            The resulting ket as a one-hot vector (or tensor product thereof).
    """
    if isinstance(args, int):
        arg_list = [args]
    else:
        arg_list = list(args)
    if isinstance(dim, int):
        dim_list = [dim] * len(arg_list)
    else:
        dim_list = list(dim)
    if len(dim_list) != len(arg_list):
        raise ValueError(f"Length mismatch: dim has {len(dim_list)} elements, "
                         f"args has {len(arg_list)} elements.")
    vecs = []
    for d_i, j_i in zip(dim_list, arg_list):
        if not (0 <= j_i < d_i):
            raise ValueError(f"Index {j_i} out of range for dimension {d_i}.")
        v = np.zeros(d_i, dtype=float)
        v[j_i] = 1.0
        vecs.append(v)
    result = vecs[0]
    for v in vecs[1:]:
        result = np.kron(result, v)
    return result

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.1', 3)
target = targets[0]

assert np.allclose(ket(2, 0), target)
target = targets[1]

assert np.allclose(ket(2, [1,1]), target)
target = targets[2]

assert np.allclose(ket([2,3], [0,1]), target)
