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
def partial_trace(X, sys, dim):
    n = len(dim)
    keep = [i for i in range(n) if i not in sys]
    perm = keep + sys
    Xp = syspermute(X, perm, dim) #
    dims_keep = [dim[i] for i in keep]
    dims_trace = [dim[i] for i in sys]
    shape = dims_keep + dims_trace + dims_keep + dims_trace
    Xp = Xp.reshape(shape)
    for _ in range(len(dims_trace)):
        axis1 = len(dims_keep)
        axis2 = len(dims_keep) + len(dims_trace)
        Xp = np.trace(Xp, axis1=axis1, axis2=axis2) #
        dims_trace = dims_trace[1:]
    D = int(np.prod(dims_keep)) if dims_keep else 1
    return Xp.reshape((D, D))
def entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 0, None)
    en = -np.sum(eigvals[eigvals > 0] * np.log(eigvals[eigvals > 0]))
    return float(en)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.6', 3)
target = targets[0]

rho = np.eye(4)/4
assert np.allclose(entropy(rho), target)
target = targets[1]

rho = np.ones((3,3))/3
assert np.allclose(entropy(rho), target)
target = targets[2]

rho = np.diag([0.8,0.2])
assert np.allclose(entropy(rho), target)
