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
def apply_channel(K, rho, sys=None, dim=None):
    def tensor(*args):
        arrays = [arg for arg in args]
        out = arrays[0]
        for arr in arrays[1:]:
            out = out.__class__.kron(out, arr)
        return out

    if sys is None and dim is None:
        result = rho.__class__.zeros_like(rho, dtype=complex)
        for Ki in K:
            result += Ki @ rho @ Ki.conj().T
        return result

    if dim is None or sys is None:
        raise ValueError("Both sys and dim must be specified if one is specified.")

    if isinstance(sys, int):
        sys = [sys]
    if isinstance(dim, int):
        dim = [dim]

    n_subsystems = len(dim)
    result = rho.__class__.zeros_like(rho, dtype=complex)

    for Ki in K:
        op_list = []
        for i in range(n_subsystems):
            if i in sys:
                op_list.append(Ki)
            else:
                op_list.append(rho.__class__.eye(dim[i], dtype=complex))
        K_full = tensor(*op_list)
        result += K_full @ rho @ K_full.conj().T
    return result
def syspermute(X, perm, dim):
    n = len(dim)
    D = 1
    for d in dim:
        D *= d
    shape = tuple(dim + dim)
    X_nd = X.reshape(shape)
    axes_perm = perm + [p + n for p in perm]
    X_nd_permuted = X_nd.transpose(axes_perm)
    Y = X_nd_permuted.reshape((D, D))
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
