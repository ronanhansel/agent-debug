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
def syspermute(X, perm, dim):
    X = [list(map(float, row)) for row in X]
    if not X or any(len(row) != len(X) for row in X):
        raise ValueError("`X` must be a square 2-D array.")
    if not isinstance(dim, (list, tuple)) or not all(isinstance(d, int) and d > 0 for d in dim):
        raise ValueError("`dim` must be a list/tuple of positive integers.")
    if not isinstance(perm, (list, tuple)):
        raise ValueError("`perm` must be a list/tuple.")
    n = len(dim)
    if len(perm) != n or sorted(perm) != list(range(n)):
        raise ValueError("`perm` must be a permutation of range(len(dim)).")
    d_tot = 1
    for d in dim:
        d_tot *= d
    if len(X) != d_tot:
        raise ValueError("Dimension mismatch: prod(dim) must equal size of `X`.")
    def idx_to_digits(idx, bases):
        digits = [0] * len(bases)
        for k in range(len(bases) - 1, -1, -1):
            digits[k] = idx % bases[k]
            idx //= bases[k]
        return digits
    def digits_to_idx(digits, bases):
        idx = 0
        for d, base in zip(digits, bases):
            idx = idx * base + d
        return idx
    dim_perm = [dim[p] for p in perm]
    Y = [[0.0 for _ in range(d_tot)] for _ in range(d_tot)]
    for i in range(d_tot):
        digits_i = idx_to_digits(i, dim)
        perm_digits_i = [digits_i[p] for p in perm]
        new_i = digits_to_idx(perm_digits_i, dim_perm)
        row_X = X[i]
        row_Y = Y[new_i]
        for j in range(d_tot):
            digits_j = idx_to_digits(j, dim)
            perm_digits_j = [digits_j[p] for p in perm]
            new_j = digits_to_idx(perm_digits_j, dim_perm)
            row_Y[new_j] = row_X[j]
    return Y

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.4', 3)
target = targets[0]

X = np.kron(np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]]))
assert np.allclose(syspermute(X, [2,1], [2,2]), target)
target = targets[1]

X = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
assert np.allclose(syspermute(X, [2,1], [2,2]), target)
target = targets[2]

X = np.kron(np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]),np.array([[1,0],[0,0]]))
assert np.allclose(syspermute(X, [1,3,2], [2,2,2]), target)
