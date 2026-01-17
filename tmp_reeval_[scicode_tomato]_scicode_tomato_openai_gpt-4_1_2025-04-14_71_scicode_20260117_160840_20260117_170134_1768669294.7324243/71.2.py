import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, args):
    def basis(d, j):
        v = [0.0] * d
        v[j] = 1.0
        return v

    if isinstance(dim, int):
        if isinstance(args, int):
            out = basis(dim, args)
        else:
            vectors = [basis(dim, j) for j in args]
            result = vectors[0]
            for vec in vectors[1:]:
                result = [a*b for a in result for b in vec]
            out = result
    else:
        vectors = [basis(d, j) for d, j in zip(dim, args)]
        result = vectors[0]
        for vec in vectors[1:]:
            result = [a*b for a in result for b in vec]
        out = result
    return out
def tensor(*args):
    arrays = [arg for arg in args]
    out = arrays[0]
    for arr in arrays[1:]:
        out = out.kron(out, arr)  #
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return out

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.2', 3)
target = targets[0]

assert np.allclose(tensor([0,1],[0,1]), target)
target = targets[1]

assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)
target = targets[2]

assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
