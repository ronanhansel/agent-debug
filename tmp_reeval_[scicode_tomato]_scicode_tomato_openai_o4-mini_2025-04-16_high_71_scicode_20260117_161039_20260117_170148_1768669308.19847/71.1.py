import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, args):
    '''Input:
    dim: int or list of ints, dimension(s) of the ket space
    args: int or list of ints, index(es) of the basis vector(s)
    Output:
    out: numpy array of float, the (possibly tensor-product) basis ket
    '''
    if isinstance(dim, int) and isinstance(args, int):
        dims = [dim]
        idxs = [args]
    elif isinstance(dim, int) and isinstance(args, (list, tuple)):
        dims = [dim] * len(args)
        idxs = list(args)
    elif isinstance(dim, (list, tuple)) and isinstance(args, (list, tuple)):
        dims = list(dim)
        idxs = list(args)
        if len(dims) != len(idxs):
            raise ValueError("Length of dim list must match length of args list.")
    else:
        raise TypeError("dim and args must each be either int or list/tuple of ints.")
    vecs = []
    for d_i, j_i in zip(dims, idxs):
        if not (isinstance(d_i, int) and isinstance(j_i, int)):
            raise TypeError("Each dimension and index must be an integer.")
        if j_i < 0 or j_i >= d_i:
            raise ValueError(f"Index {j_i} out of range for dimension {d_i}.")
        e = np.zeros(d_i, dtype=float)
        e[j_i] = 1.0
        vecs.append(e)
    out = vecs[0]
    for v in vecs[1:]:
        out = np.kron(out, v)
    return out

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.1', 3)
target = targets[0]

assert np.allclose(ket(2, 0), target)
target = targets[1]

assert np.allclose(ket(2, [1,1]), target)
target = targets[2]

assert np.allclose(ket([2,3], [0,1]), target)
