import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, args):
    # Helper to decide whether an object is sequence-like (but not a string)
    is_seq = lambda obj: isinstance(obj, (list, tuple))
    # Convert inputs to lists for uniform processing
    dims = list(dim) if is_seq(dim) else [int(dim)]
    idxs = list(args) if is_seq(args) else [int(args)]
    # If a single dimension is provided for multiple indices, replicate it
    if len(dims) == 1 and len(idxs) > 1:
        dims *= len(idxs)
    if len(dims) != len(idxs):
        raise ValueError("Length mismatch between 'dim' and 'args'.")
    # Build individual basis vectors |j_k⟩ of length d_k
    basis_vectors = []
    for d_k, j_k in zip(dims, idxs):
        if not (0 <= j_k < d_k):
            raise ValueError(f"Index {j_k} out of range for dimension {d_k}.")
        v = [0.0] * d_k
        v[j_k] = 1.0
        basis_vectors.append(v)
    # Tensor (Kronecker) product of all subsystem basis vectors
    def kron(a, b):
        return [ai * bj for ai in a for bj in b]
    out = basis_vectors[0]
    for v in basis_vectors[1:]:
        out = kron(out, v)
    return out
def tensor(*args):
    '''
    Takes the tensor product of an arbitrary number of matrices/vectors.
    
    Input
    -----
    args : any number of nd arrays of floats
        Each argument is a matrix or vector (array-like) to include in the
        Kronecker product.
    
    Output
    ------
    M : numpy.ndarray
        The tensor (Kronecker) product of all input arrays.  If every input is
        one-dimensional the result is a 1-D array; otherwise it is at least
        two-dimensional.
    '''
    if len(args) == 0:
        raise ValueError("tensor() requires at least one input array.")
    M = np.asarray(args[0], dtype=float)
    for factor in args[1:]:
        M = np.kron(M, np.asarray(factor, dtype=float))
    return M

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.2', 3)
target = targets[0]

assert np.allclose(tensor([0,1],[0,1]), target)
target = targets[1]

assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)
target = targets[2]

assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
