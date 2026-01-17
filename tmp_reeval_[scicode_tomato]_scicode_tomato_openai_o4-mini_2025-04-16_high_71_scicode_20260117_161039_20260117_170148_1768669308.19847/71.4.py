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
def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (Kronecker product) of input matrices, 2d array of floats
    '''
    if len(args) == 0:
        raise ValueError("tensor() requires at least one array argument.")
    M = np.array(args[0], dtype=float)
    for arr in args[1:]:
        A = np.array(arr, dtype=float)
        M = np.kron(M, A)
    return M
def apply_channel(K, rho, sys=None, dim=None):
    '''Applies the channel with Kraus operators in K to the state rho on
    the subsystem(s) specified by sys. The dimensions of the subsystems of rho
    are given by dim. If sys and dim are both None, the channel acts on the
    entire system.
    Inputs:
      K   : list of 2D arrays (float or complex), the Kraus operators
      rho : 2D array (float or complex), the input density matrix
      sys : int or list of int, target subsystem index(es); None → full system
      dim : int or list of int, dimension(s) of each subsystem; None → single system
    Output:
      2D array: the output density matrix after applying the channel
    '''
    if sys is None and dim is None:
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho must be a square matrix.")
        dim_list = [rho.shape[0]]
        sys_list = [0]
    elif (sys is None) ^ (dim is None):
        raise ValueError("sys and dim must be provided together, or both None.")
    else:
        sys_list = [sys] if isinstance(sys, int) else list(sys)
        dim_list = [dim] if isinstance(dim, int) else list(dim)
        for d in dim_list:
            if not (isinstance(d, int) and d > 0):
                raise ValueError("Each entry in dim must be a positive integer.")
        for s in sys_list:
            if not (isinstance(s, int) and 0 <= s < len(dim_list)):
                raise ValueError(f"Subsystem index {s} out of range.")
    total_dim = 1
    for d in dim_list:
        total_dim *= d
    if rho.shape != (total_dim, total_dim):
        raise ValueError("rho shape does not match product of dim entries.")
    if len(sys_list) != 1:
        raise NotImplementedError("Only single‐subsystem channels (len(sys)==1) supported.")
    target = sys_list[0]
    sub_dim = dim_list[target]
    for Ki in K:
        Ki_arr = np.array(Ki, dtype=complex)
        if Ki_arr.ndim != 2 or Ki_arr.shape[0] != Ki_arr.shape[1]:
            raise ValueError("Each Kraus operator must be a square matrix.")
        if Ki_arr.shape[0] != sub_dim:
            raise ValueError("Kraus operator dimension does not match target subsystem.")
    use_complex = np.iscomplexobj(rho) or any(np.iscomplexobj(Ki) for Ki in K)
    dtype = complex if use_complex else float
    output = np.zeros((total_dim, total_dim), dtype=dtype)
    for Ki in K:
        Ki_arr = np.array(Ki, dtype=dtype)
        ops = []
        for i, d in enumerate(dim_list):
            if i == target:
                ops.append(Ki_arr)
            else:
                ops.append(np.eye(d, dtype=dtype))
        K_ext = ops[0]
        for A in ops[1:]:
            K_ext = np.kron(K_ext, A)
        output += K_ext @ rho @ K_ext.conj().T
    return output
def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
      X   : 2d array of floats (or complex) with shape (D, D), where
            D = product(dim)
      perm: list of int containing a permutation of [0, 1, ..., n−1]
      dim : list of int containing the dimensions of all n subsystems
    Output:
      Y   : 2d array of floats (or complex) with shape (D, D),
            the density matrix of the permuted state
    '''
    n = len(dim)
    if len(perm) != n or set(perm) != set(range(n)):
        raise ValueError("perm must be a permutation of [0, 1, ..., n-1].")
    if any(not (isinstance(d, int) and d > 0) for d in dim):
        raise ValueError("Each entry in dim must be a positive integer.")
    X_arr = np.array(X)
    total_dim = 1
    for d in dim:
        total_dim *= d
    if X_arr.ndim != 2 or X_arr.shape != (total_dim, total_dim):
        raise ValueError("X must be square with shape matching the product of dim entries.")
    reshaped = X_arr.reshape(tuple(dim) + tuple(dim))
    new_axes = perm + [p + n for p in perm]
    permuted = reshaped.transpose(new_axes)
    Y = permuted.reshape((total_dim, total_dim))
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
