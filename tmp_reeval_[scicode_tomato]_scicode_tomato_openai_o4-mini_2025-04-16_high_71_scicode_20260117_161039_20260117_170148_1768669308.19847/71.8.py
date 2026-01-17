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
def partial_trace(X, sys, dim):
    """Inputs:
    X   : 2d square array (float or complex), shape (D,D) with D = prod(dim)
    sys : list of int, indices of subsystems to trace out (discard)
    dim : list of int, dimensions of all subsystems
    Output:
    2d array (float or complex), the reduced density matrix after tracing out sys
    """
    X = np.array(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix.")
    if not isinstance(dim, (list, tuple)) or len(dim) < 1:
        raise ValueError("dim must be a non-empty list or tuple of positive ints.")
    if any((not isinstance(d, int)) or d <= 0 for d in dim):
        raise ValueError("Each entry in dim must be a positive integer.")
    n = len(dim)
    if not isinstance(sys, (list, tuple)):
        raise TypeError("sys must be a list or tuple of subsystem indices.")
    sys_list = list(sys)
    if any((not isinstance(s, int)) or s < 0 or s >= n for s in sys_list):
        raise ValueError("Each subsystem index in sys must be in [0, len(dim)).")
    if len(set(sys_list)) != len(sys_list):
        raise ValueError("Subsystem indices in sys must be unique.")
    total_dim = 1
    for d in dim:
        total_dim *= d
    if X.shape != (total_dim, total_dim):
        raise ValueError("Shape of X does not match product of dim entries.")
    keep = [i for i in range(n) if i not in sys_list]
    perm_first = keep + sys_list
    Xp = syspermute(X, perm_first, dim)
    dim_perm = [dim[i] for i in perm_first]
    rho = Xp.reshape(tuple(dim_perm) + tuple(dim_perm))
    k = len(keep)
    t = len(sys_list)
    m = rho.ndim // 2
    for j in reversed(range(t)):
        rho = np.trace(rho, axis1=k + j, axis2=m + k + j)
    dim_keep = [dim[i] for i in keep]
    if not dim_keep:
        return np.array([[rho]])
    D_keep = 1
    for d in dim_keep:
        D_keep *= d
    return rho.reshape((D_keep, D_keep))
def entropy(rho):
    rho_arr = np.array(rho, dtype=complex)
    if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_arr.shape[1]:
        raise ValueError("rho must be a square matrix.")
    evals = np.linalg.eigvalsh(rho_arr)
    evals = np.clip(np.real(evals), 0, None)
    lam = evals[evals > 0]
    en = -np.sum(lam * np.log(lam))
    return float(np.real(en))
def generalized_amplitude_damping_channel(gamma, N):
    """
    Generates the generalized amplitude damping channel.
    Inputs:
      gamma: float, damping parameter (0 ≤ gamma ≤ 1)
      N    : float, thermal parameter (0 ≤ N ≤ 1)
    Output:
      kraus: list of four 2×2 NumPy arrays of floats, [K1, K2, K3, K4]
    """
    gamma = float(gamma)
    N = float(N)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be between 0 and 1.")
    if not (0.0 <= N <= 1.0):
        raise ValueError("N must be between 0 and 1.")

    a = np.sqrt(1.0 - N)
    b = np.sqrt(1.0 - gamma)
    c = np.sqrt(gamma * (1.0 - N))
    d = np.sqrt(N)
    e = np.sqrt(gamma * N)

    P00 = np.array([[1.0, 0.0],
                    [0.0, 0.0]], dtype=float)
    P11 = np.array([[0.0, 0.0],
                    [0.0, 1.0]], dtype=float)
    P01 = np.array([[0.0, 1.0],
                    [0.0, 0.0]], dtype=float)
    P10 = np.array([[0.0, 0.0],
                    [1.0, 0.0]], dtype=float)

    K1 = a * (P00 + b * P11)
    K2 = c * P01
    K3 = d * (b * P00 + P11)
    K4 = e * P10

    return [K1, K2, K3, K4]
def neg_rev_coh_info(p, g, N):
    psi = np.zeros(4, dtype=complex)
    psi[0] = np.sqrt(1.0 - p)
    psi[3] = np.sqrt(p)
    rho_in = np.outer(psi, psi.conj())
    kraus_ops = generalized_amplitude_damping_channel(g, N)
    I2 = np.eye(2, dtype=complex)
    rho_out = np.zeros((4, 4), dtype=complex)
    for K in kraus_ops:
        Kc = np.array(K, dtype=complex)
        K_ext = np.kron(I2, Kc)
        temp = np.dot(rho_in, K_ext.conj().T)
        rho_out += np.dot(K_ext, temp)
    rho_A = partial_trace(rho_out, sys=[1], dim=[2, 2])
    S_AB = entropy(rho_out)
    S_A = entropy(rho_A)
    return float(S_AB - S_A)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.8', 3)
target = targets[0]

p = 0.477991
g = 0.2
N = 0.4
assert np.allclose(neg_rev_coh_info(p,g,N), target)
target = targets[1]

p = 0.407786
g = 0.2
N = 0.1
assert np.allclose(neg_rev_coh_info(p,g,N), target)
target = targets[2]

p = 0.399685
g = 0.4
N = 0.2
assert np.allclose(neg_rev_coh_info(p,g,N), target)
