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
def apply_channel(K, rho, sys=None, dim=None):
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("`rho` must be a square 2-D array (density matrix).")
    d_tot = rho.shape[0]
    def _tensor(*arrays):
        out = np.asarray(arrays[0], dtype=float)
        for arr in arrays[1:]:
            out = np.kron(out, np.asarray(arr, dtype=float))
        return out
    if sys is None or dim is None:
        out = np.zeros_like(rho, dtype=float)
        for Ki in K:
            Ki = np.asarray(Ki, dtype=float)
            if Ki.shape != (d_tot, d_tot):
                raise ValueError("Kraus operator dimension mismatch.")
            out += Ki @ rho @ Ki.T
        return out
    if not isinstance(sys, (list, tuple)) or len(sys) != 1:
        raise NotImplementedError("Only single-subsystem channels supported.")
    if not isinstance(dim, (list, tuple)):
        raise ValueError("`dim` must be a list/tuple when `sys` is provided.")
    if np.prod(dim) != d_tot:
        raise ValueError("Product of `dim` must equal the dimension of `rho`.")
    tgt = sys[0]
    if not (0 <= tgt < len(dim)):
        raise ValueError("Subsystem index out of range.")
    d_tgt = dim[tgt]
    identities = [np.eye(d, dtype=float) for d in dim]
    out = np.zeros((d_tot, d_tot), dtype=float)
    for Ki in K:
        Ki = np.asarray(Ki, dtype=float)
        if Ki.shape != (d_tgt, d_tgt):
            raise ValueError("Kraus operator incompatible with target dimension.")
        factors = identities.copy()
        factors[tgt] = Ki
        G = _tensor(*factors)
        out += G @ rho @ G.T
    return out

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.3', 3)
target = targets[0]

K = [np.eye(2)]
rho = np.array([[0.8,0],[0,0.2]])
assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)
target = targets[1]

K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
rho = np.ones((2,2))/2
assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)
target = targets[2]

K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
assert np.allclose(apply_channel(K, rho, sys=[2], dim=[2,2]), target)
